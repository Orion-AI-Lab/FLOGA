'''
Visualizes predictions of a specified model.
'''

import argparse
import numpy as np
from pathlib import Path
import pyjson5
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset_utils import Dataset
from utils import (
    font_colors,
    resume_or_start,
    init_model,
    validate_configs
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', required=True, help='The path to save results into.')
    parser.add_argument('--config', type=str, default='configs/config.json', required=False,
                        help='The config file to use. Default "configs/config.json".')
    parser.add_argument('--bands', type=str, choices=['rgb', 'nrg'], default='nrg', required=False,
                        help='If "rgb" is given, then R-G-B bands are plotted. If "nrg" is given, then NIR-R-G bands are plotted. Default "nrg".')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], default='test', required=False,
                        help='The dataset to produce predictions for. Default "test".')
    parser.add_argument('--export_numpy_preds', action='store_true', default=False, required=False,
                        help='Also export the model predictions in a separate numpy array.')

    args = parser.parse_args()

    # Check argument
    assert Path(args.config).exists(), \
        print(f'{font_colors.RED}{font_colors.BOLD}The given config file ({args.config}) does not exist!{font_colors.ENDC}')

    # Read config file
    configs = pyjson5.load(open(args.config, 'r'))
    model_configs = pyjson5.load(open(Path('configs') / 'method' / f'{configs["method"]}.json', 'r'))

    # Validate config file
    validate_configs(configs)

    # Set mode to 'eval'
    configs['mode'] = 'eval'

    # Set paths
    run_path, resume_from_checkpoint, init_epoch = resume_or_start(configs, model_configs)

    results_path = Path(args.results_path)

    split_mode = args.mode
    results_path = results_path / split_mode
    results_path.mkdir(exist_ok=True, parents=True)

    # Create separate folders for TP, FP, FN images
    fp_path = results_path / 'FP'
    fp_path.mkdir(exist_ok=True, parents=True)

    tp_path = results_path / 'TP'
    tp_path.mkdir(exist_ok=True, parents=True)

    fn_path = results_path / 'FN'
    fn_path.mkdir(exist_ok=True, parents=True)

    # Get device
    # TODO: Add support for multiple gpus
    if len(configs['gpu_ids']) == 1:
        device = f'cuda:{configs["gpu_ids"][0]}'
    else:
        device = 'cpu'

    # Print informative message
    print(f'{font_colors.CYAN}--- Loading model checkpoint: {configs["paths"]["load_state"]} ---{font_colors.ENDC}')

    # Load checkpoint
    checkpoint = torch.load(resume_from_checkpoint, map_location=device)

    # Get data sources and GSDs
    tmp = configs['dataset_type'].split('_')
    # format: "sen2_xx_mod_yy"
    gsd = {tmp[0]: tmp[1], tmp[2]: tmp[3]}

    # Update the configs with the SEN2 or MODIS bands to be used
    data_source = configs['datasets']['data_source']
    for band in configs['datasets']['selected_bands'][data_source].keys():
        configs['datasets']['selected_bands'][data_source][band] = configs['datasets'][f'{data_source}_bands'][gsd[data_source]][band]

    # Compute total number of input channels
    inp_channels = len(configs['datasets']['selected_bands'][data_source])

    dataset = Dataset(args.mode, configs)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Initialize model
    model = init_model(configs, model_configs, checkpoint, inp_channels, device, run_path=run_path)

    model = model[0]
    model.eval()

    if args.bands == 'rgb':
        # Get band indices for R, G, B
        if data_source == 'sen2':
            plot_bands = [
                configs['datasets']['selected_bands']['sen2']['B04'],
                configs['datasets']['selected_bands']['sen2']['B03'],
                configs['datasets']['selected_bands']['sen2']['B02']
            ]
            selected_bands_idx = {
                'B04': configs['datasets']['selected_bands']['sen2']['B04'],
                'B03': configs['datasets']['selected_bands']['sen2']['B03'],
                'B02': configs['datasets']['selected_bands']['sen2']['B02']
            }
        else:
            plot_bands = [
                configs['datasets']['selected_bands']['mod']['B01'],
                configs['datasets']['selected_bands']['mod']['B04'],
                configs['datasets']['selected_bands']['mod']['B03']
            ]
            selected_bands_idx = {
                'B01': configs['datasets']['selected_bands']['mod']['B01'],
                'B04': configs['datasets']['selected_bands']['mod']['B04'],
                'B03': configs['datasets']['selected_bands']['mod']['B03']
            }
    else:
        # Get band indices for NIR, R, G
        bands = configs['datasets']['selected_bands'][data_source]
        selected_bands_idx = {band: order_id for order_id, (band, _) in enumerate(bands.items())}

        if data_source == 'sen2':
            if set(['B08', 'B04', 'B03']) <= set(configs['datasets']['selected_bands']['sen2'].keys()):
                # NIR, Red, Green
                plot_bands = [selected_bands_idx[band] for band in ['B08', 'B04', 'B03']]
            else:
                # NIR, Red, Green
                plot_bands = [selected_bands_idx[band] for band in ['B8A', 'B04', 'B03']]
        else:
            if set(['B02', 'B01', 'B04']) <= configs['datasets']['selected_bands']['mod'].keys():
                # NIR, Red, Green
                plot_bands = [selected_bands_idx[band] for band in ['B02', 'B01', 'B04']]
            else:
                # NIR, Red, Green
                plot_bands = [selected_bands_idx[band] for band in ['B02', 'B01']]

    # Define custom colormap for the labels
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', [(0, 0, 0, 10), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0), (0.8647058823529412, 0.30980392156862746, 0.45882352941176474, 1.0)], 3)
    title_font = {'size': '22'}

    nplots = 4

    with tqdm(initial=0, total=len(loader)) as pbar:
        for image_id, batch in enumerate(loader):
            plot_i = 0
            with torch.no_grad():

                bands = list(configs['datasets']['selected_bands'][data_source].values())

                if data_source == 'mod':
                    before_img = batch['MOD_before_image'][:, bands, :, :].to(device)
                    after_img = batch['MOD_after_image'][:, bands, :, :].to(device)
                    label = batch['label'].to(device).long()
                else:
                    before_img = batch['S2_before_image'][:, bands, :, :].to(device)
                    after_img = batch['S2_after_image'][:, bands, :, :].to(device)
                    label = batch['label'].to(device).long()

                if configs['method'] == 'changeformer':
                    output = model(before_img, after_img)
                    output = output[-1]
                    predictions = output.argmax(1).to(dtype=torch.int8)
                else:
                    output = model(before_img, after_img)
                    predictions = output.argmax(1).to(dtype=torch.int8)

                before_img = before_img.squeeze()
                after_img = after_img.squeeze()
                label = label.squeeze()

                fig, axes = plt.subplots(1, nplots, figsize=(80, 10), num=1, clear=True)

                before_img = before_img[plot_bands, :, :].detach().cpu().numpy()
                before_img = np.clip(before_img, a_min=0, a_max=1)
                axes[plot_i].imshow(np.moveaxis(before_img, 0, -1))

                axes[plot_i].set_title(f'S-2 Before ({image_id})', fontdict=title_font)
                axes[plot_i].set_xticks([])
                axes[plot_i].set_yticks([])
                axes[plot_i].spines['top'].set_visible(False)
                axes[plot_i].spines['right'].set_visible(False)
                axes[plot_i].spines['bottom'].set_visible(False)
                axes[plot_i].spines['left'].set_visible(False)
                plot_i += 1

                after_img = after_img[plot_bands, :, :].detach().cpu().numpy()
                after_img = np.clip(after_img, a_min=0, a_max=1)
                axes[plot_i].imshow(np.moveaxis(after_img, 0, -1))

                axes[plot_i].set_title(f'S-2 After ({image_id})', fontdict=title_font)
                axes[plot_i].set_xticks([])
                axes[plot_i].set_yticks([])
                axes[plot_i].spines['top'].set_visible(False)
                axes[plot_i].spines['right'].set_visible(False)
                axes[plot_i].spines['bottom'].set_visible(False)
                axes[plot_i].spines['left'].set_visible(False)
                plot_i += 1

                label = label.squeeze().cpu().detach().numpy()
                axes[plot_i].imshow(label, vmin=0, vmax=2, cmap=cmap)
                axes[plot_i].set_title(f'S-2 Label ({image_id})', fontdict=title_font)
                axes[plot_i].set_xticks([])
                axes[plot_i].set_yticks([])
                axes[plot_i].spines['top'].set_visible(False)
                axes[plot_i].spines['right'].set_visible(False)
                axes[plot_i].spines['bottom'].set_visible(False)
                axes[plot_i].spines['left'].set_visible(False)
                plot_i += 1

                predictions = predictions.squeeze().cpu().detach().numpy()
                axes[plot_i].imshow(predictions, vmin=0, vmax=2, cmap=cmap)
                axes[plot_i].set_title(f'S-2 Prediction ({image_id})', fontdict=title_font)
                axes[plot_i].set_xticks([])
                axes[plot_i].set_yticks([])
                axes[plot_i].spines['top'].set_visible(False)
                axes[plot_i].spines['right'].set_visible(False)
                axes[plot_i].spines['bottom'].set_visible(False)
                axes[plot_i].spines['left'].set_visible(False)
                plot_i += 1


                sample_id = batch['key'][0]

                if np.any(predictions == 1):
                    if np.any(label == 1):
                        plt.savefig(tp_path / f'prediction_visualization_ID{image_id}_{sample_id}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
                    else:
                        plt.savefig(fp_path / f'prediction_visualization_ID{image_id}_{sample_id}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
                elif np.any(label == 1):
                    plt.savefig(fn_path / f'prediction_visualization_ID{image_id}_{sample_id}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
                else:
                    plt.savefig(results_path / f'prediction_visualization_ID{image_id}_{sample_id}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

                if args.export_numpy_preds:
                    np.save(results_path / f'ID{image_id}_{sample_id}.npy', predictions)

            pbar.update(1)