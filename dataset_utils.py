from pathlib import Path
import numpy as np
import pandas as pd
import random
import pickle
import h5py

from einops import rearrange
import torch
from torch.utils.data import Sampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# Seed stuff
np.random.seed(999)
random.seed(999)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, configs, clc=False, clouds=False, sea=False):
        self.mode = mode
        self.configs = configs

        self.augmentation = configs['datasets']['augmentation']

        self.ds_path = Path(configs['paths']['dataset']) / configs['dataset_type']

        # Read the pickle files containing information on the splits
        patches = pickle.load(open(self.ds_path / configs['datasets'][mode], 'rb'))

        self.events_df = pd.DataFrame([{**{'sample_key': k}, **patches[k]} for k in sorted(list(patches.keys()))])

        # Keep the positive indices in a separate list (useful for under/oversampling)
        self.positives_idx = list(self.events_df[self.events_df['positive_flag']]['sample_key'].values)

        # format: "sen2_xx_mod_yy"
        tmp = configs['dataset_type'].split('_')
        self.gsd = {tmp[0]: tmp[1], tmp[2]: tmp[3]}

        self.clc = clc
        self.sea = sea
        self.clouds = clouds

        self.selected_bands = {}
        self.means = {}
        self.stds = {}
        for k, v in self.gsd.items():
            self.selected_bands[k] = configs['datasets']['selected_bands'][k].values()
            self.means[k] = [m for i, m in enumerate(configs['datasets'][f'{k}_mean'][v]) if i in self.selected_bands[k]]
            self.stds[k] = [m for i, m in enumerate(configs['datasets'][f'{k}_std'][v]) if i in self.selected_bands[k]]


    def scale_img(self, sample):
        '''
        Scales the given images with the method defined in the config file.
        The input `sample` is a dictionary mapping image name -> image array.
        '''
        scaled_sample = sample.copy()

        for sample_name, sample_img in sample.items():
            if ('label' in sample_name) or ('cloud' in sample_name) or ('key' in sample_name) or ('positive' in sample_name) or ('sea' in sample_name) or ('clc' in sample_name):
                scaled_sample[sample_name] = sample_img
            elif self.configs['datasets']['scale_input'] == 'normalize':
                if 'S2' in sample_name:
                    scaled_sample[sample_name] = TF.normalize(sample_img, mean=self.means['sen2'], std=self.stds['sen2'])
                elif 'MOD' in sample_name:
                    scaled_sample[sample_name] = TF.normalize(sample_img, mean=self.means['mod'], std=self.stds['mod'])
            elif self.configs['datasets']['scale_input'] == 'min-max':
                mins = sample_img.min(dim=-1).values.min(dim=-1).values
                maxs = sample_img.max(dim=-1).values.max(dim=-1).values

                uniq_mins = mins.unique()
                uniq_maxs = maxs.unique()
                if not (((len(uniq_mins) == 1) and (uniq_mins.item() == 0.)) and ((len(uniq_maxs) == 1) and (uniq_maxs.item() == 0.))):
                    # Some images are all-zeros so scaling returns a NaN image
                    new_ch = []
                    for ch in range(sample_img.shape[0]):
                        if mins[ch] == maxs[ch]:
                            # Some channels contain only a single value, so scaling returns all-NaN
                            # We convert it to all-zeros
                            new_ch.append(torch.zeros(*sample_img[ch, :, :].shape)[None, :, :])
                        else:
                            new_ch.append(((sample_img[ch, :, :] - mins[:, None, None][ch]) / (maxs[:, None, None][ch] - mins[:, None, None][ch]))[None, :, :])

                    scaled_sample[sample_name] = torch.cat(new_ch, dim=0)
            elif isinstance(self.configs['datasets']['scale_input'], list):
                new_min, new_max = [torch.tensor(i) for i in self.configs['datasets']['scale_input']]

                mins = sample_img.min(dim=-1).values.min(dim=-1).values
                maxs = sample_img.max(dim=-1).values.max(dim=-1).values

                uniq_mins = mins.unique()
                uniq_maxs = maxs.unique()
                if not (((len(uniq_mins) == 1) and (uniq_mins.item() == 0.)) and ((len(uniq_maxs) == 1) and (uniq_maxs.item() == 0.))):
                    # Some images are all-zeros so scaling returns a NaN image
                    new_ch = []
                    for ch in range(sample_img.shape[0]):
                        if mins[ch] == maxs[ch]:
                            # Some channels contain only a single value, so scaling returns all-NaN
                            # We convert it to all-zeros
                            new_ch.append(torch.zeros(*sample_img[ch, :, :].shape)[None, :, :])
                        else:
                            new_ch.append(((sample_img[ch, :, :] - mins[:, None, None][ch]) / (maxs[:, None, None][ch] - mins[:, None, None][ch]))[None, :, :])

                    scaled_sample[sample_name] = torch.mul(torch.cat(new_ch, dim=0), (new_max - new_min)) + new_min
            elif self.configs['datasets']['scale_input'].startswith('clamp_scale'):
                thresh = int(self.configs['datasets']['scale_input'].split('_')[-1])
                scaled_sample[sample_name] = torch.clamp(sample_img, max=thresh)
                scaled_sample[sample_name] = scaled_sample[sample_name] / thresh
            elif self.configs['datasets']['scale_input'].startswith('clamp'):
                thresh = int(self.configs['datasets']['scale_input'].split('_')[-1])
                scaled_sample[sample_name] = torch.clamp(sample_img, max=thresh)

        return scaled_sample


    def load_img(self, sample):
        '''
        Loads the images associated with a single event. The input `sample` is a list of filenames for
        the event.

        Returns a dictionary mapping image name -> image array.
        '''
        loaded_sample = {}

        for sample_info in sample.index:
            if sample_info == 'sample_key':
                loaded_sample['key'] = sample[sample_info]
            elif sample_info == 'positive_flag':
                loaded_sample['positive'] = sample[sample_info]
            elif ('label' in sample_info):
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]))
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info])
            elif self.clouds and ('cloud' in sample_info):
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]).astype(np.float32))
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info])
            elif 'sea' in sample_info:
                if self.sea:
                    if sample[sample_info].suffix == '.npy':
                        loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]))
                    else:
                        loaded_sample[sample_info] = torch.load(sample[sample_info])
            elif 'S2' in sample_info:
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]).astype(np.float32)).to(torch.float32)
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info]).to(torch.float32)
            elif 'MOD' in sample_info:
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]).astype(np.float32)).to(torch.float32)
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info]).to(torch.float32)
            elif self.clc and ('clc' in sample_info):
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]).astype(np.float32))
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info])

        return loaded_sample


    def fillna(self, sample):
        '''
        Fills NaN values in the sample with the constant specified in the config.

        It also replaces the corresponding values in the label with the number '2' which will be ignored during training.
        '''
        filled_sample = sample.copy()

        nan_idx = []
        label = []
        for sample_name, s in sample.items():
            if 'label' in sample_name:
                label.append(sample_name)
            elif ('cloud' in sample_name) or ('clc' in sample_name):
                continue
            elif ('before' in sample_name) or ('after' in sample_name):
                nan_idx.append(torch.isnan(s))
                filled_sample[sample_name] = torch.nan_to_num(s, nan=self.configs['datasets']['nan_value'])

        for lbl in label:
            for nan_id in nan_idx:
                for band_id in nan_id:
                    filled_sample[lbl][band_id] = 2

        return filled_sample


    def augment(self, sample):
        '''
        Applies the following augmentations:
        - Random horizontal flipping (possibility = 0.5)
        - Random vertical flipping (possibility = 0.5)
        - Random rotation (-15 to +15 deg)
        '''
        aug_sample = sample.copy()

        # Horizontal flip
        if random.random() > 0.5:
            for sample_name, s in aug_sample.items():
                if sample_name not in ['key', 'positive']:
                    aug_sample[sample_name] = TF.hflip(s)

        # Vertical flip
        if random.random() > 0.5:
            for sample_name, s in aug_sample.items():
                if sample_name not in ['key', 'positive']:
                    aug_sample[sample_name] = TF.vflip(s)

        # Rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            for sample_name, s in aug_sample.items():
                if sample_name not in ['key', 'positive']:
                    if s.dim() == 2:
                        # For some reason `TF.rotate()` cannot handle 2D input
                        aug_sample[sample_name] = TF.rotate(torch.unsqueeze(s, 0), angle=angle).squeeze()
                    else:
                        aug_sample[sample_name] = TF.rotate(s, angle=angle)

        return aug_sample


    def __len__(self):
        return self.events_df.shape[0]


    def __getitem__(self, event_id):
        batch = self.events_df.iloc[event_id]

        # Load images
        batch = self.load_img(batch)

        # Replace NaN values with constant
        batch = self.fillna(batch)

        # Normalize images
        if self.configs['datasets']['scale_input'] is not None:
            batch = self.scale_img(batch)

        # Augment images
        if self.augmentation:
            batch = self.augment(batch)

        return batch


class OverSampler(Sampler):
    '''
    A Sampler which performs oversampling in imbalanced datasets.
    '''
    def __init__(self, dataset, positive_prc=0.5):
        self.dataset = dataset
        self.positive_prc = positive_prc
        self.n_samples = len(dataset)


    def __iter__(self):
        positives = self.dataset.events_df[self.dataset.events_df['positive_flag']].index.values
        pos = np.random.choice(positives, int(self.positive_prc * self.n_samples), replace=True)
        neg = np.random.choice(list(set(self.dataset.events_df.index.values) - set(positives)), int(((1 - self.positive_prc) * self.n_samples) + 1))

        idx = np.hstack([pos, neg])
        np.random.shuffle(idx)

        idx = idx[:self.n_samples]

        pos_cnt = len([i for i in idx if i in pos])
        print(f'Using {pos_cnt} POS and {len(idx) - pos_cnt} NEG (1:{((len(idx) - pos_cnt) / pos_cnt):.2f}).')

        return iter(idx)


    def __len__(self):
        return len(self.dataset)