'''
This script creates an analysis-ready dataset from the downloaded FLOGA imagery.
'''

from pathlib import Path
import argparse
import numpy as np
import h5py
import tqdm
import pickle
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F

from utils import font_colors, SEN2_BORDER, NODATA_VALUES


def export_patches(floga_path, out_path, out_size, out_format, sea_ratio):
    '''
    Loads the FLOGA HDF files and exports the cropped patches into separate folders (one folder per year of data).
    '''
    for hdf_file in floga_path.glob('*.hdf'):
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create analysis-ready dataset from FLOGA imagery.')

    parser.add_argument('--floga_path', type=str, required=True,
                        help='The path containing the FLOGA HDF files.')
    parser.add_argument('--out_path', type=str, required=True,
                        help='The path to save the dataset into. Default: "data/datasets/"')
    parser.add_argument('--out_format', type=str, choices=['numpy', 'torch'], default='numpy', required=False,
                        help='The output format, can be either "numpy" or "torch". Default "numpy".')
    parser.add_argument('--out_size', nargs='+', required=False,
                        help='Output images will be cropped to this size, if given.')
    parser.add_argument('--stages', default='1 2', nargs='+', required=False,
                        help='The preprocessing stages to execute:\n' + \
                        'Stage 1: Export cropped samples\n' + \
                        'Stage 2: Split train/val/test sets (and sample if needed)\n')
    parser.add_argument('--sea_ratio', type=float, default=0.9, required=False,
                        help='The sea:land ratio to identify sea patches with. These patches are removed. Default 0.9.')
    parser.add_argument('--split_mode', type=str, choices=['year', 'event'], default='event', required=False,
                        help='How to split the train/val/test sets in Stage 3. If "year" then --train_years and --test_years must ' + \
                        'also be defined. If "event" then all years are used. Default "event".')
    parser.add_argument('--ratio', nargs='+', type=int, default=[60, 20, 20], required=False,
                        help='The train/val/test ratio for Stage 3. Default "60 20 20".')
    parser.add_argument('--train_years', nargs='+', default=[], required=False,
                        help='The years to read data for in order to create the train/validation sets in Stage 3 if --split_mode=="year".')
    parser.add_argument('--test_years', nargs='+', default=[], required=False,
                        help='The years to read data for in order to create the test set in Stage 3 if --split_mode=="year".')
    parser.add_argument('--sample', type=int, required=False,
                        help='A number indicating the ratio of negative patches to sample. It corresponds to 1:<sample> for positives:negatives.')
    parser.add_argument('--num_workers', type=int, default=4, required=False,
                        help='The number of workers to use. Default 4.')
    parser.add_argument('--suffix', type=str, required=False,
                        help='A suffix to use for the exported pickles containing information on the data split. Default None.')
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
        export_patches(floga_path, out_path, out_size, args.out_format, args.sea_ratio)

    # ---- STAGE 2 ----
    # Split train/val/test sets

    if '2' in args.stages:
        pass