'''
This script is used to train a model on the given data. The user must
specify the necessary data paths, the model and its hyperparameters.
'''

import argparse
from pathlib import Path
import pyjson5
from tqdm import tqdm
import copy
import wandb
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import (
    font_colors,
    resume_or_start,
    init_model,
    compute_class_weights,
    validate_configs
)
from dataset_utils import Dataset, OverSampler
from cd_experiments_utils import (
    train_change_detection,
    eval_change_detection,
)


def init_wandb(model, run_path, configs, model_configs):
    if configs['wandb']['resume_wandb']:
        # Resume wandb from stored id
        wandb_id = pyjson5.load(open(run_path / 'wandb_id.json', 'r'))['run_id']
    else:
        # Create new wandb id
        wandb_id = wandb.util.generate_id()
        pyjson5.dump({"run_id": wandb_id}, open(run_path / 'wandb_id.json', 'wb'), quote_keys=True)

    all_configs = copy.deepcopy(configs)
    all_configs['model_configs'] = model_configs

    wandb.init(project=configs['wandb']['wandb_project'], entity=configs['wandb']['wandb_entity'], config=all_configs, id=wandb_id, resume="allow")

    wandb.watch(model[0], log_freq=20)

    return wandb


def get_positive_sample(mode, ds, configs):
    '''
    Finds and returns the id assigned to the first positive sample in the given dataset.
    '''
    candidate_paths = [i for i in Path(configs['paths']['dataset']).glob('*') if i.name == configs['dataset_type']]

    ds_path = candidate_paths[0]

    patches = pickle.load(open(ds_path / configs['datasets'][mode], 'rb'))
    first_positive = [k for k, v in patches.items() if v['positive_flag']][0]

    return ds.events_df[ds.events_df['sample_key'] == first_positive].index.values[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config.json', required=False,
                        help='The config file to use. Default "configs/config.json".')

    args = parser.parse_args()

    # Check argument
    assert Path(args.config).exists(), \
        print(f'{font_colors.RED}{font_colors.BOLD}The given config file ({args.config}) does not exist!{font_colors.ENDC}')

    # Read config file
    configs = pyjson5.load(open(args.config, 'r'))
    model_configs = pyjson5.load(open(Path('configs') / 'method' / f'{configs["method"]}.json', 'r'))

    # Validate config file
    validate_configs(configs)

    # Set paths
    run_path, resume_from_checkpoint, init_epoch = resume_or_start(configs, model_configs)

    # Print informative message
    if configs['mode'] == 'train':
        if (resume_from_checkpoint is None) or ((isinstance(resume_from_checkpoint, list)) and not any(resume_from_checkpoint)):
            print(f'{font_colors.CYAN}--- Training a new model ---{font_colors.ENDC}')
            print(f'{font_colors.CYAN}--- Model path: {run_path} ---{font_colors.ENDC}')
        else:
            print(f'{font_colors.CYAN}--- Resuming training from {resume_from_checkpoint} ---{font_colors.ENDC}')

            if not configs['train']['resume']:
                print(f'{font_colors.CYAN}--- New model path: {run_path} ---{font_colors.ENDC}')
    else:
        print(f'{font_colors.CYAN}--- Testing model for {resume_from_checkpoint} ---{font_colors.ENDC}')

    # Get device
    # TODO: Add support for multiple GPUs
    if len(configs['gpu_ids']) == 1:
        device = f'cuda:{configs["gpu_ids"][0]}'
    else:
        device = 'cpu'

    # Load checkpoint
    if configs['mode'] == 'train':
        if resume_from_checkpoint is not None:
            checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        else:
            checkpoint = resume_from_checkpoint

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

    # Compute class weights for the specific dataset
    class_weights = compute_class_weights(configs)

    configs['paths']['run_path'] = run_path

    if configs['mode'] == 'train':
        # --- Train model ---

        # Initialize datasets and dataloaders
        train_dataset = Dataset('train', configs)
        val_dataset = Dataset('val', configs, clc=True)

        if isinstance(configs['datasets']['oversampling'], float):
            train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True, sampler=OverSampler(train_dataset, positive_prc=configs['datasets']['oversampling']))
        else:
            train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

        # Get a validation image id for visualization
        validation_id = get_positive_sample('val', val_dataset, configs)

        print(f'{font_colors.CYAN}Using {configs["train"]["loss_function"]} with class weights: {class_weights["train"]}.{font_colors.ENDC}')

        # Begin training
        for rep_i, _ in enumerate(range(configs['train']['rep_times'])):
            # Initialize model
            model = init_model(configs, model_configs, checkpoint, inp_channels, device, run_path=run_path, init_epoch=init_epoch)

            # Initialize wandb
            if configs['wandb']['activate'] and (rep_i == 0):
                wandb = init_wandb(model, run_path, configs, model_configs)

            if len(model) == 1:
                model = model[0]

            train_change_detection(model, device, class_weights, run_path, init_epoch, train_loader, val_loader, validation_id,
                                   gsd, checkpoint, configs, model_configs, rep_i, wandb)

    # --- Test model ---

    # Begin testing
    results = {'f1': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'lc_stats': []}

    # Initialize datasets and dataloaders
    test_dataset = Dataset('test', configs, clc=True)

    validation_id = get_positive_sample('test', test_dataset, configs)

    test_loader = DataLoader(test_dataset, batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True, num_workers=configs['datasets']['num_workers'])

    # Initialize model
    model = init_model(configs, model_configs, None, inp_channels, device, run_path=run_path, init_epoch=init_epoch)
    if len(model) == 1:
        model = model[0]

    for rep_i, _ in enumerate(range(configs['train']['rep_times'])):
        # Initialize wandb
        if (configs['mode'] == 'eval') and configs['wandb']['activate'] and (rep_i == 0):
            wandb = init_wandb(model, run_path, configs, model_configs)

        ckpt_path = run_path / 'checkpoints' / f'{rep_i}' / 'best_segmentation.pt'
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f'\n{font_colors.CYAN}Loading {ckpt_path}...{font_colors.ENDC}')

        model.load_state_dict(checkpoint['model_state_dict'])
        res = eval_change_detection(model, device, class_weights, init_epoch, test_loader, validation_id,
                                    gsd, 'test', configs, model_configs, rep_i, wandb, run_path)

        for k, v in res.items():
            results[k].append(v)

    # Print final results
    print('\n ===============\n')

    for k, v in results.items():
        if k == 'lc_stats':
            continue
        else:
            print(f'{k} (burnt): {round(np.mean([i[1] for i in v]), 2)} ({round(np.std([i[1] for i in v]), 2)})')
            print(f'{k} (unburnt): {round(np.mean([i[0] for i in v]), 2)} ({round(np.std([i[0] for i in v]), 2)})')

    mean_f1 = np.mean([np.mean([i[0] for i in results['f1']]), np.mean([i[1] for i in results['f1']])])
    mean_f1_std = np.mean([np.std([i[0] for i in results['f1']]), np.std([i[1] for i in results['f1']])])
    print(f'Mean f-score: {round(mean_f1, 2)} ({round(mean_f1_std, 2)})')

    mean_iou = np.mean([np.mean([i[0] for i in results['iou']]), np.mean([i[1] for i in results['iou']])])
    mean_iou_std = np.mean([np.std([i[0] for i in results['iou']]), np.std([i[1] for i in results['iou']])])
    print(f'Mean IoU: {round(mean_iou, 2)} ({round(mean_iou_std, 2)})')

    if configs['train']['log_landcover_metrics']:
        # Print statistics for each land cover type
        import pandas as pd

        df = pd.DataFrame(columns=['Land cover type', 'Accuracy', 'Precision (0)', 'Precision (1)', 'Recall (0)', 'Recall (1)',
                'F1-score (0)', 'F1-score (1)', 'IoU (0)', 'IoU (1)', 'mF1', 'mIoU'])

        print('')

        lc_ids = [list(i.keys()) for i in results['lc_stats']]
        lc_ids = np.unique([item for sublist in lc_ids for item in sublist])

        for lc_id in lc_ids:
            acc = np.mean([i[lc_id]['acc'] for i in results["lc_stats"] if lc_id in i.keys()])
            acc_std = np.std([i[lc_id]['acc'] for i in results["lc_stats"] if lc_id in i.keys()])
            prec_0 = np.mean([i[lc_id]['prec'][0] for i in results["lc_stats"] if lc_id in i.keys()])
            prec_0_std = np.std([i[lc_id]['prec'][0] for i in results["lc_stats"] if lc_id in i.keys()])
            prec_1 = np.mean([i[lc_id]['prec'][1] for i in results["lc_stats"] if lc_id in i.keys()])
            prec_1_std = np.std([i[lc_id]['prec'][1] for i in results["lc_stats"] if lc_id in i.keys()])
            rec_0 = np.mean([i[lc_id]['rec'][0] for i in results["lc_stats"] if lc_id in i.keys()])
            rec_0_std = np.std([i[lc_id]['rec'][0] for i in results["lc_stats"] if lc_id in i.keys()])
            rec_1 = np.mean([i[lc_id]['rec'][1] for i in results["lc_stats"] if lc_id in i.keys()])
            rec_1_std = np.std([i[lc_id]['rec'][1] for i in results["lc_stats"] if lc_id in i.keys()])
            f1_0 = np.mean([i[lc_id]['f1'][0] for i in results["lc_stats"] if lc_id in i.keys()])
            f1_0_std = np.std([i[lc_id]['f1'][0] for i in results["lc_stats"] if lc_id in i.keys()])
            f1_1 = np.mean([i[lc_id]['f1'][1] for i in results["lc_stats"] if lc_id in i.keys()])
            f1_1_std = np.std([i[lc_id]['f1'][1] for i in results["lc_stats"] if lc_id in i.keys()])
            iou_0 = np.mean([i[lc_id]['iou'][0].item() for i in results["lc_stats"] if lc_id in i.keys()])
            iou_0_std = np.std([i[lc_id]['iou'][0].item() for i in results["lc_stats"] if lc_id in i.keys()])
            iou_1 = np.mean([i[lc_id]['iou'][1].item() for i in results["lc_stats"] if lc_id in i.keys()])
            iou_1_std = np.std([i[lc_id]['iou'][1].item() for i in results["lc_stats"] if lc_id in i.keys()])

            name = [i[lc_id]['name'] for i in results["lc_stats"] if lc_id in i.keys()][0]

            df = pd.concat(
                [
                    df,
                    pd.DataFrame({
                    'Land cover type': [name],
                    'Accuracy': [f'{round(acc, 2)} ({round(acc_std, 2)})'],
                    'Precision (0)': [f'{round(prec_0, 2)} ({round(prec_0_std, 2)})'],
                    'Precision (1)': [f'{round(prec_1, 2)} ({round(prec_1_std, 2)})'],
                    'Recall (0)': [f'{round(rec_0, 2)} ({round(rec_0_std, 2)})'],
                    'Recall (1)': [f'{round(rec_1, 2)} ({round(rec_1_std, 2)})'],
                    'F1-score (0)': [f'{round(f1_0, 2)} ({round(f1_0_std, 2)})'],
                    'F1-score (1)': [f'{round(f1_1, 2)} ({round(f1_1_std, 2)})'],
                    'IoU (0)': [f'{round(iou_0, 2)} ({round(iou_0_std, 2)})'],
                    'IoU (1)': [f'{round(iou_1, 2)} ({round(iou_1_std, 2)})'],
                    'mF1': [f'{round(np.nanmean([f1_0, f1_1]), 2)} ({round(np.nanmean([f1_0_std, f1_1_std]), 2)})'],
                    'mIoU': [f'{round(np.nanmean([iou_0, iou_1]).item(), 2)} ({round(np.nanmean([iou_0_std, iou_1_std]), 2)}']
                    })],
                ignore_index = True
            )

            df.to_csv(run_path / 'lc_stats.csv', index=False)