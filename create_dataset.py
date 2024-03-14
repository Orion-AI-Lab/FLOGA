'''
This script creates an analysis-ready dataset from the downloaded FLOGA imagery.
'''

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from tqdm import tqdm
import shutil
import pickle
import copy
from itertools import product
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F

from utils import font_colors


SEN2_BORDER = 65535
NODATA_VALUES = {
    'SEN2': 0,
    'MOD': -28672
}


def get_padding_offset(img_size, out_size):
    img_size_x = img_size[0]
    img_size_y = img_size[1]

    output_size_x = out_size[0]
    output_size_y = out_size[1]

    # Calculate padding offset
    if img_size_x >= output_size_x:
        pad_x = int(output_size_x - img_size_x % output_size_x)
    else:
        # For bigger images, is just the difference
        pad_x = output_size_x - img_size_x

    if img_size_y >= output_size_y:
        pad_y = int(output_size_y - img_size_y % output_size_y)
    else:
        # For bigger images, is just the difference
        pad_y = output_size_y - img_size_y

    # Number of rows that need to be padded (top and bot)
    if not pad_x == output_size_x:
        pad_top = int(pad_x // 2)
        pad_bot = int(pad_x // 2)

        # if padding is not equally divided, pad +1 row to the top
        if not pad_x % 2 == 0:
            pad_top += 1
    else:
        pad_top = 0
        pad_bot = 0

    # Number of rows that need to be padded (left and right)
    if not pad_y == output_size_y:
        pad_left = int(pad_y // 2)
        pad_right = int(pad_y // 2)

        # if padding is not equally divided, pad +1 row to the left
        if not pad_y % 2 == 0:
            pad_left += 1
    else:
        pad_left = 0
        pad_right = 0

    return pad_top, pad_bot, pad_left, pad_right


def pad_image(img, out_size, pad_top, pad_bot, pad_left, pad_right):
    '''
    If the given image cannot be evenly divided into the required subpatch dimensions,
    then a padded version is returned.
    '''
    # Check if padding is needed
    if ((img.shape[0] % out_size[0]) != 0) or ((img.shape[1] % out_size[1]) != 0):
        img = np.pad(img,
                     pad_width=((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
                     mode='constant',
                     constant_values=0)

    return img


def export_patches(floga_path, out_path, out_size, out_format, sea_ratio):
    '''
    Loads the FLOGA HDF files and exports the cropped patches into separate folders (one folder per year of data).
    '''
    zero_patches = []
    cloudy_patches = []
    other_burnt_areas_patches = []
    sea_patches = []
    recorded = []

    if out_format == 'torch':
        img_fmt = '*.pt'
    elif out_format == 'numpy':
        img_fmt = '*.npy'

    hdf_files = list(floga_path.glob('*.h5'))

    with tqdm(initial=0, total=len(hdf_files)) as pbar:
        for hdf_file_i, hdf_file in enumerate(hdf_files):
            pbar.set_description(f'({hdf_file_i + 1}/{len(hdf_files)}) {hdf_file.name}')

            hdf = h5py.File(hdf_file, 'r')
            year, _, sen_gsd, _, _ = hdf_file.stem.split('_')[2:]

            image_names = [
                'clc_100_mask',
                'label',
                'mod_500_cloud_post',
                'mod_500_cloud_pre',
                'mod_500_post',
                'mod_500_pre',
                'sea_mask',
                f'sen2_{sen_gsd}_cloud_post',
                f'sen2_{sen_gsd}_cloud_pre',
                f'sen2_{sen_gsd}_post',
                f'sen2_{sen_gsd}_pre'
            ]

            out_path_hdf = out_path / year
            out_path_hdf.mkdir(parents=True, exist_ok=True)

            for event_id, event_imgs in hdf[year].items():
                # Compute padding offsets for the images of this event
                img = event_imgs['label'][:].squeeze()

                padding_offsets = get_padding_offset(img.shape, out_size)

                # Get the required indices to split the image
                if out_size is not None:
                    x_idx = list(range(0, img.shape[0] + padding_offsets[0] + padding_offsets[1], out_size[0]))
                    y_idx = list(range(0, img.shape[1] + padding_offsets[2] + padding_offsets[3], out_size[1]))
                else:
                    x_idx = [0]
                    y_idx = [0]
                    out_size = [img.shape[0], img.shape[1]]

                crop_indices = [x_idx, y_idx]

                # Get the number of patches contained in this event's files
                number_of_patches = len(crop_indices[0]) * len(crop_indices[1])

                # Split the files of the current event and insert them to the dataset as a single patch
                # i.e. a patch contains the same cropped subregion of each file in this event
                # and it has a single id
                for img_name in image_names:
                    img = event_imgs[img_name][:]

                    patch_offset = 0
                    
                    # Expand the channels dimension if image is 2D (simplifies calculations later)
                    if img.ndim == 2:
                        img = img[None, :, :]

                    if ('label' not in img_name) and ('mask' not in img_name):
                        # Replace 'nodata' value with 0
                        if 'sen2' in img_name:
                            img[img == NODATA_VALUES['SEN2']] = 0
                        elif 'mod' in img_name:
                            img[img == NODATA_VALUES['MOD']] = 0
                    elif img_name == 'sea_mask':
                        # Convert to binary mask
                        img[(img == 2) | (img == 4)] = 0
                        img[img != 0] = 1

                    # Pad image
                    img = pad_image(img, out_size, *padding_offsets)

                    if out_format == 'torch':
                        img = torch.from_numpy(img)

                    # Split image
                    for x, y in product(*crop_indices):
                        patch_key = f'sample{(patch_offset):08d}_{event_id}_{year}'
                        patch_offset += 1

                        with open(out_path_hdf / 'sample_indices.txt', 'a') as f:
                            if patch_key not in recorded:
                                f.write(f'{patch_key}: ({x}, {y})\n')
                                recorded.append(patch_key)

                        if img_name == 'label':
                            # If current sample image is the label, add an indicator for negative/positive
                            if isinstance(img, np.ndarray):
                                uniqs = np.unique(img[:, x:x+out_size[0], y:y+out_size[1]])
                            else:
                                uniqs = img[:, x:x+out_size[0], y:y+out_size[1]].unique()

                            # Add an indicator file for positive/negative samples
                            if 1 in uniqs:
                                with open(out_path_hdf / f'{patch_key}.positive.label', 'w') as f:
                                    f.write('')
                            else:
                                with open(out_path_hdf / f'{patch_key}.negative.label', 'w') as f:
                                    f.write('')

                            # If label contains the value '2', then mark it for removal
                            # This value represents burnt areas of other events in the same year
                            if isinstance(img, np.ndarray):
                                if np.any(img == 2):
                                    other_burnt_areas_patches.append(patch_key)
                            elif torch.any(img == 2).item():
                                other_burnt_areas_patches.append(patch_key)
                        elif img_name == 'clc_100_mask':
                            pass
                        elif 'cloud' in img_name:
                            # If current image is a cloud mask, mark cloudy patches for removal
                            if 'sen2' in img_name:
                                # NOTE: We set the 'cloudy' pixels to be more than 2 because Sentinel-2 cloud mask
                                # often contains arbitrary 'cloud' pixels
                                if isinstance(img, np.ndarray):
                                    if np.where(img[0, x:x+out_size[0], y:y+out_size[1]] == 9)[0].shape[0] > 10:
                                        cloudy_patches.append(patch_key)
                                elif torch.where(img[0, x:x+out_size[0], y:y+out_size[1]] == 9)[0].shape[0] > 10:
                                    cloudy_patches.append(patch_key)
                            # NOTE: the MODIS QC flags do not seem to correctly catch clouds so we ignore them
                        elif img_name == 'sea_mask':
                            # If patch contains only sea/water, mark it for removal
                            if isinstance(img, np.ndarray):
                                uniqs, uniqs_c = np.unique(img[:, x:x+out_size[0], y:y+out_size[1]], return_counts=True)
                                if (1 in uniqs) and (uniqs_c[uniqs == 1][0] >= (sea_ratio * out_size[0] * out_size[1])):
                                    sea_patches.append(patch_key)
                            else:
                                uniqs, uniqs_c = img[0, x:x+out_size[0], y:y+out_size[1]].unique(return_counts=True)
                                if (1 in uniqs) and (uniqs_c[uniqs == 1].item() >= (sea_ratio * out_size[0] * out_size[1])):
                                    sea_patches.append(patch_key)
                        else:
                            # If current image is a surface reflectance and it contains all zeros, then mark this patch for removal
                            # NOTE: We check just a single channel for zeros
                            if isinstance(img, np.ndarray):
                                uniqs, counts = np.unique(img[0, x:x+out_size[0], y:y+out_size[1]], return_counts=True)
                                if (0 in uniqs) and (counts[np.where(uniqs == 0)] == (out_size[0] * out_size[1])):
                                    zero_patches.append(patch_key)
                            else:
                                uniqs, counts = torch.unique(img[0, x:x+out_size[0], y:y+out_size[1]], return_counts=True)
                                if (0 in uniqs) and (counts[torch.where(uniqs == 0)].item() == (out_size[0] * out_size[1])):
                                    zero_patches.append(patch_key)

                        # Crop image
                        if out_format == 'numpy':
                            np.save(out_path_hdf / f'{patch_key}.{img_name}.npy', img[:, x:x+out_size[0], y:y+out_size[1]].squeeze().copy())
                        else:
                            torch.save(img[:, x:x+out_size[0], y:y+out_size[1]].squeeze(), out_path_hdf / f'{patch_key}.{img_name}.pt')

            # Delete patches with zeros (border patches) and
            # move sea/water patches to another folder
            burnt, unburnt = 0, 0
            sea_patches_dir = out_path / f'{year}_SEA'
            sea_patches_dir.mkdir(parents=True, exist_ok=True)
            for f in out_path.glob('*'):
                if f.stem.split('.')[0] in zero_patches:
                    # If current patch is marked for removal, delete it
                    f.unlink()
                if (f.stem.split('.')[0] in sea_patches) and f.exists():
                    # If current patch is marked as sea/water patch, move it
                    shutil.move(f, sea_patches_dir)

            # Export zero, cloudy and sea/water patch names to files
            with open(out_path / 'zero_patches.txt', 'a') as f:
                for patch_key in set(zero_patches):
                    f.write(f'{patch_key}\n')

            with open(out_path / 'cloudy_patches.txt', 'a') as f:
                for patch_key in set(cloudy_patches):
                    f.write(f'{patch_key}\n')

            with open(out_path / 'sea_patches.txt', 'a') as f:
                for patch_key in set(sea_patches):
                    f.write(f'{patch_key}\n')

            # Log useful information on patch selection
            with open(out_path / 'logs', 'a') as f:
                f.write(f'HDF file: {hdf_file.name}\n')
                f.write('Deleted:\n')
                f.write(f'   - {len(set(zero_patches))} all-zero patches\n')
                f.write(f'   - {len(set(cloudy_patches))} cloudy patches\n')
                f.write('Moved:\n')
                f.write(f'   - {len(set(sea_patches))} sea patches with sea ratio >= {sea_ratio}\n')
                f.write(f'Patches with burnt areas of other events: {len(other_burnt_areas_patches)}\n\n')

            pbar.update(1)

    return


def export_csv_with_patch_paths(events_list, out_path, mode, random_seed, split_mode, ratio, train_years, test_years, suffix=None, sampling=None):
    '''
    Exports the paths of the selected events into pickle files, one file per split (train/val/test).
    Each pickle file contains a dictionary of the form e.g.:
        {
            patch_key: {
                label: <path to label>,
                positive_flag: <path to flag>,
                S2_pre_image: <path to Sentinel-2 pre-event image>,
                S2_post_image: <path to Sentinel-2 post-event image>,
                ...
            },
        }
    In the case of a per-year split, the name of the pickle contains information on the train/test years:
        "{train_years}_{test_years}_{mode}{_suffix}.pkl"
    Otherwise, the name of the exported pickle file has the following form:
        "allEvents_{mode}{_suffix}.pkl".
    '''
    if suffix is None:
        suffix = ''
    else:
        suffix = f'_{suffix}'

    dct = {}
    for event in events_list:
        event_id, year = event.split('_')

        # Find patches corresponding to this event
        event_patches = (out_path / f'{year}').glob(f'*_{event_id}_{year}.*')

        # Group patches by key
        patch_files = {}
        for f in event_patches:
            k = f.name.split('.')[0]
            if k not in patch_files.keys():
                patch_files[k] = [f]
            else:
                patch_files[k] += [f]

        for patch_key, files in patch_files.items():
            dct[patch_key] = {}
            for f in files:
                if 'sea_mask' in f.name:
                    dct[patch_key]['sea_mask'] = f
                elif 'clc_100_mask' in f.name:
                    dct[patch_key]['clc_mask'] = f
                elif 'mod' in f.name:
                    if ('cloud' in f.name) and ('pre' in f.name):
                        dct[patch_key]['MOD_before_cloud'] = f
                    elif ('cloud' in f.name) and ('post' in f.name):
                        dct[patch_key]['MOD_after_cloud'] = f
                    elif 'pre' in f.name:
                        dct[patch_key]['MOD_before_image'] = f
                    elif 'post' in f.name:
                        dct[patch_key]['MOD_after_image'] = f
                elif 'sen2' in f.name:
                    if ('cloud' in f.name) and ('pre' in f.name):
                        dct[patch_key]['S2_before_cloud'] = f
                    elif ('cloud' in f.name) and ('post' in f.name):
                        dct[patch_key]['S2_after_cloud'] = f
                    elif 'pre' in f.name:
                        dct[patch_key]['S2_before_image'] = f
                    elif 'post' in f.name:
                        dct[patch_key]['S2_after_image'] = f
                if 'label' in f.stem:
                    dct[patch_key]['label'] = f
                elif f.suffix == '.label':
                    if 'positive' in f.name:
                        dct[patch_key]['positive_flag'] = True
                    else:
                        dct[patch_key]['positive_flag'] = False

    if sampling is not None:
        dct = sample_patches(sampling, dct, random_seed)
        sampling_str = f'r{sampling}'
    else:
        sampling_str = ''

    ratio_str = '-'.join([str(i) for i in ratio])

    if split_mode == 'year':
        pickle.dump(dct, open(out_path / f'{"".join(train_years)}_{"".join(test_years)}_{ratio_str}_{sampling_str}{suffix}_{mode}.pkl', 'wb'))
    else:
        pickle.dump(dct, open(out_path / f'allEvents_{ratio_str}_{sampling_str}{suffix}_{mode}.pkl', 'wb'))


def sample_patches(neg_ratio, d, random_seed):
    '''
    Randomly selects a number of negative samples and updates the given dictionary accordingly.
    The number of selected samples is defined by the given `ratio` argument, which corresponds to
    the ratio of positives:negatives, aka 1:`ratio`.
    '''
    d_sampled = copy.deepcopy(d)

    negatives = []
    positives = []
    for patch_key, patch_images in d.items():
        if patch_images['positive_flag']:
            positives.append(patch_key)
        else:
            negatives.append(patch_key)

    num_negatives = len(positives) * neg_ratio

    # If the number of positive samples surpasses the number of negative samples, then nothing is done
    if len(positives) >= len(negatives): return d

    rng = np.random.default_rng(random_seed)
    selected_negatives = rng.choice(len(negatives), size=num_negatives, replace=False)
    for selected_idx in range(len(negatives)):
        if selected_idx in selected_negatives: continue
        del d_sampled[negatives[selected_idx]]

    return d_sampled


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create analysis-ready dataset from FLOGA imagery.')

    parser.add_argument('--floga_path', type=str, required=True,
                        help='The path containing the FLOGA HDF files.')
    parser.add_argument('--out_path', type=str, required=True,
                        help='The path to save the dataset into. Default: "data/datasets/"')
    parser.add_argument('--out_format', type=str, choices=['numpy', 'torch'], default='numpy', required=False,
                        help='The output format, can be either "numpy" or "torch". Default "numpy".')
    parser.add_argument('--out_size', nargs='+', required=False,
                        help='Width and height separated by space. Output images will be cropped to this size, if given.')
    parser.add_argument('--stages', default='1 2', nargs='+', required=False,
                        help='The preprocessing stages to execute:\n' + \
                        'Stage 1: Export cropped samples\n' + \
                        'Stage 2: Split train/val/test sets (and sample if needed)\n')
    parser.add_argument('--sea_ratio', type=float, default=0.9, required=False,
                        help='The sea:land ratio to identify sea patches with. These patches are removed. Default 0.9.')
    parser.add_argument('--split_mode', type=str, choices=['year', 'event'], default='event', required=False,
                        help='How to split the train/val/test sets in Stage 2. If "year" then --train_years and --test_years must ' + \
                        'also be defined. If "event" then all years are used. Default "event".')
    parser.add_argument('--ratio', nargs='+', type=int, default=[60, 20, 20], required=False,
                        help='The train/val/test ratio for Stage 2. Default "60 20 20".')
    parser.add_argument('--train_years', nargs='+', default=[], required=False,
                        help='The years to read data for in order to create the train/validation sets in Stage 2 if --split_mode=="year" (separated by space).')
    parser.add_argument('--test_years', nargs='+', default=[], required=False,
                        help='The years to read data for in order to create the test set in Stage 2 if --split_mode=="year" (separated by space).')
    parser.add_argument('--sample', type=int, required=False,
                        help='A number indicating the ratio of negative patches to sample in Stage 2. It corresponds to 1:<sample> for positives:negatives.')
    parser.add_argument('--num_workers', type=int, default=4, required=False,
                        help='The number of workers to use. Default 4.')
    parser.add_argument('--suffix', type=str, required=False,
                        help='A suffix to use for the exported pickle files containing information on the data split. Default None.')
    parser.add_argument('--random_seed', type=int, required=False,
                        help='A random seed to use for reproducible results.')

    args = parser.parse_args()

    # Set up paths
    floga_path = Path(args.floga_path)
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.out_size is not None:
        out_size = [int(i) for i in args.out_size]
    else:
        out_size = None

    # ---- STAGE 1 ----
    # Export the images cropped into dimensions of (`out_size` x `out_size`) without overlap

    if '1' in args.stages:
        print(f'\n{font_colors.CYAN}--- STAGE 1 ---{font_colors.ENDC}\n')
        export_patches(floga_path, out_path, out_size, args.out_format, args.sea_ratio)

    # ---- STAGE 2 ----
    # Split train/val/test sets

    if '2' in args.stages:
        print(f'\n{font_colors.CYAN}--- STAGE 2 ---{font_colors.ENDC}\n')

        # Check arguments
        if args.split_mode == 'year':
            assert (args.train_years is not None) and (args.test_years is not None), \
                print(f'{font_colors.RED}Error: You must provide the train years and test years!{font_colors.ENDC}')

            assert set(args.train_years).isdisjoint(set(args.test_years)), \
                print(f'{font_colors.RED}Error: The given train and test years overlap!{font_colors.ENDC}')

        else:
            assert (args.ratio is not None) and all([isinstance(i, int) for i in args.ratio]) and (sum(args.ratio) == 100), \
                print(f'{font_colors.RED}Error: You must provide a valid ratio for data splitting!{font_colors.ENDC}')

        if args.out_format == 'torch':
            img_fmt = '*.pt'
        elif args.out_format == 'numpy':
            img_fmt = '*.npy'

        if args.split_mode == 'year':
            train_events_list = []
            for year in args.train_years:
                for f in (out_path / year).glob(f'*.{img_fmt}'):
                    f_event, f_year = f.name.split('.')[0].split('_')[1:3]
                    if f'{f_event}_{f_year}' not in train_events_list:
                        train_events_list.append(f'{f_event}_{f_year}')

            test_events_list = []
            for year in args.test_years:
                for f in (out_path / year).glob(f'*.{img_fmt}'):
                    f_event, f_year = f.name.split('.')[0].split('_')[1:3]
                    if f'{f_event}_{f_year}' not in test_events_list:
                        test_events_list.append(f'{f_event}_{f_year}')

            print(f'{font_colors.CYAN}Found {len(train_events_list)} train/val and {len(test_events_list)} test events to export.{font_colors.ENDC}')

            print(f'{font_colors.CYAN}\nSplitting train/val/test sets...{font_colors.ENDC}')

            labels_list = [0] * len(train_events_list)
            train_events_list, val_events_list, _, _ = train_test_split(train_events_list, labels_list, train_size=args.ratio[0] / 100, random_state=args.random_seed)
        else:
            # Find all available data
            all_events = []
            for d in out_path.glob('*'):
                for f in d.glob('*.label'):
                    all_events.append('_'.join(f.stem.split('.')[0].split('_')[1:]))

            all_events = list(set(all_events))

            print(f'{font_colors.CYAN}Found {len(all_events)} events to export.{font_colors.ENDC}')

            print(f'{font_colors.CYAN}\nSplitting train/val/test sets...{font_colors.ENDC}')

            # Split train/val - test
            labels_list = [0] * len(all_events)
            trainval_events_list, test_events_list, _, _ = train_test_split(all_events, labels_list, train_size=(args.ratio[0] + args.ratio[1]) / 100, random_state=args.random_seed)

            # Split train - val
            labels_list = [0] * len(trainval_events_list)
            new_ratio = (args.ratio[0] * 100) / (args.ratio[0] + args.ratio[1])
            train_events_list, val_events_list, _, _ = train_test_split(trainval_events_list, labels_list, train_size=new_ratio / 100, random_state=args.random_seed)

            print(f'{font_colors.CYAN}Splitting into {len(train_events_list)} train, {len(val_events_list)} val and {len(test_events_list)} test events...{font_colors.ENDC}')

        # Export CSV files with the paths for every split
        export_csv_with_patch_paths(train_events_list, out_path, 'train', args.random_seed, args.split_mode, args.ratio, args.train_years, args.test_years, suffix=args.suffix, sampling=args.sample)
        export_csv_with_patch_paths(val_events_list, out_path, 'val', args.random_seed, args.split_mode, args.ratio, args.train_years, args.test_years, suffix=args.suffix, sampling=args.sample)
        export_csv_with_patch_paths(test_events_list, out_path, 'test', args.random_seed, args.split_mode, args.ratio, args.train_years, args.test_years, suffix=args.suffix, sampling=args.sample)