from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pyjson5
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix, JaccardIndex

from models.fc_ef_conc import FC_EF_conc
from models.fc_ef_diff import FC_EF_diff
from models.unet import Unet
from models.snunet import SNUNet_ECAM
from models.hfanet import HFANet
from models.changeformer import ChangeFormerV6
from models.bit_cd import define_G
from models.adhr_cdnet import ADHR
from models.transunet_cd import TransUNet_CD
from models.bam_cd.model import BAM_CD

from losses.dice import DiceLoss
from losses.bce_and_dice import BCEandDiceLoss


class font_colors:
    '''
    Colors for printing messages to stdout.
    '''
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


class NotSupportedError(Exception):
    def __init__(self, method_name, argument, message=""):
        self.message = f'"{argument}" is not yet supported for {method_name}!'
        super().__init__(self.message)


def validate_configs(configs):
    '''
    Performs basic checks on the validity of the configs file.
    '''
    assert configs['method'] not in [None, ''], \
        f'{font_colors.RED}Error: No method is provided.{font_colors.ENDC}'

    assert configs['dataset_type'] not in [None, ''], \
        f'{font_colors.RED}Error: No dataset type is provided.{font_colors.ENDC}'

    assert configs['mode'] in ['train', 'eval'], \
        f'{font_colors.RED}Error: "mode" must be either "train" or "eval".{font_colors.ENDC}'

    assert configs['datasets']['data_source'] in ['mod', 'sen2'], \
        f'{font_colors.RED}Error: "data_source" must be either "mod" or "sen2".{font_colors.ENDC}'

    assert Path(configs['paths']['dataset']).exists(), \
        f'{font_colors.RED}{font_colors.BOLD}The dataset path ({configs["paths"]["dataset"]}) does not exist!{font_colors.ENDC}'
    ds_path = Path(configs['paths']['dataset'])

    if configs['paths']['load_state'] is not None:
        load_path = Path(configs['paths']['load_state']).parent
        ckpt = Path(configs['paths']['load_state']).name
        assert any([p for p in load_path.glob(f'{ckpt}*')]), \
            f'{font_colors.RED}{font_colors.BOLD}The checkpoint path ({configs["paths"]["load_state"]}) does not exist!{font_colors.ENDC}'

    candidate_paths = [i for i in ds_path.glob('*') if configs['dataset_type'] in i.name]
    if configs['mode'] == 'train':
        split_path = ds_path / configs['dataset_type'] / configs['datasets']['train']
        assert split_path.exists(), \
            f'{font_colors.RED}{font_colors.BOLD}The train dataset ({split_path}) does not exist!{font_colors.ENDC}'
        split_path = ds_path / configs['dataset_type'] / configs['datasets']['val']
        assert split_path.exists(), \
            f'{font_colors.RED}{font_colors.BOLD}The validation dataset ({split_path}) does not exist!{font_colors.ENDC}'
    else:
        split_path = ds_path / configs['dataset_type'] / configs['datasets']['test']
        assert split_path.exists(), \
            f'{font_colors.RED}{font_colors.BOLD}The test dataset ({split_path}) does not exist!{font_colors.ENDC}'


def init_model_log_path(configs, model_configs):
    '''
    Initializes the path to save results for the given model.

    The general form of a model's path is:
        "results/<task_name>/<model_name>/<timestamp>/"
    and its checkpoints are saved in:
        "results/<task_name>/<model_name>/<timestamp>/checkpoints/<ckpt_name>"
    '''
    if configs['mode'] == 'train':
        if configs['train']['resume']:
            # Use an existing path to resume training
            assert Path(configs['paths']['load_state']).exists(), \
                print(f'{font_colors.RED}Error: path {configs["paths"]["load_state"]} does not exist!.{font_colors.ENDC}')

            resume_path = Path(configs['paths']['load_state'])
            results_path = Path(*resume_path.parts[:-3])
        else:
            # Create a new path
            run_ts = datetime.now().strftime("%Y%m%d%H%M%S")
            results_path = Path(configs['paths']['results']) / configs['method'] / run_ts
            results_path.mkdir(exist_ok=True, parents=True)

            ckpt_path = results_path / 'checkpoints'
            ckpt_path.mkdir(exist_ok=True, parents=True)
    else:
        # Use an existing path for testing
        assert Path(configs['paths']['load_state']).exists(), \
            print(f'{font_colors.RED}Error: path {configs["paths"]["load_state"]} does not exist!.{font_colors.ENDC}')

        test_path = Path(configs['paths']['load_state'])
        results_path = Path(*test_path.parts[:-3])

    return results_path


def resume_or_start(configs, model_configs):
    '''
    Checks whether training must resume or start from scratch and returns
    the appropriate training parameters for each case.

    Parameters
    ----------
    configs: dict
        The configuration file.
    model_configs: dict
        The configuration file of the specific model.

    Returns
    -------
    (Path, Path | [Path], int): the path containing results from all runs,
    the path(s) containing the last checkpoint in case of resuming,
    the initial epoch.
    '''
    run_path = init_model_log_path(configs, model_configs)

    if configs['mode'] == 'train':
        # Training mode
        # See if any checkpoints are specified
        if configs['paths']['load_state'] is not None:
            load_checkpoint = Path(configs['paths']['load_state'])
            resume_from_checkpoint = load_checkpoint
        else:
            resume_from_checkpoint = None

        if configs['train']['resume']:
            # Training resumes from given checkpoint(s)
            init_epoch = int(resume_from_checkpoint.stem.split('epoch=')[1]) + 1
        else:
            # Training starts from scratch
            init_epoch = 0
    else:
        # Testing mode
        resume_from_checkpoint = Path(configs['paths']['load_state'])

        init_epoch = 0

    return run_path, resume_from_checkpoint, init_epoch


def init_model(configs, model_configs, checkpoint, inp_channels, device, run_path=None, init_epoch=None):
    models = []

    if configs['method'] == 'fc_ef_conc':
        models.append(FC_EF_conc(
            input_nbr=inp_channels,
            label_nbr=2
        ))
    elif configs['method'] == 'fc_ef_diff':
        models.append(FC_EF_diff(
            input_nbr=inp_channels,
            label_nbr=2
        ))
    elif configs['method'] == 'unet':
        models.append(Unet(
            input_nbr=inp_channels,
            label_nbr=2
        ))
    elif configs['method'] == 'snunet':
        models.append(SNUNet_ECAM(
            inp_channels,
            2,
            base_channel=model_configs['base_channel']
        ))
    elif configs['method'] == 'bam_cd':
        models.append(BAM_CD(
            encoder_name=model_configs['backbone'],
            encoder_weights=model_configs['encoder_weights'],
            in_channels=inp_channels,
            classes=2,
            fusion_mode='conc',
            activation=model_configs['activation'],
            siamese=model_configs['siamese'],
            decoder_attention_type=model_configs["decoder_attention_type"],
            decoder_use_batchnorm=model_configs['decoder_use_batchnorm']
        ))
    elif configs['method'] == 'hfanet':
        models.append(HFANet(
            input_channel=inp_channels,
            input_size=configs['datasets']['img_size'],
            num_classes=2))
    elif configs['method'] == 'bit_cd':
        models.append(define_G(
            model_configs,
            num_classes=2,
            in_channels=inp_channels))
    elif configs['method'] == 'changeformer':
        models.append(ChangeFormerV6(
            embed_dim=model_configs['embed_dim'],
            input_nc= inp_channels,
            output_nc=2,
            decoder_softmax=model_configs['decoder_softmax']))
    elif configs['method'] == 'adhr_cdnet':
        models.append(ADHR(
            in_channels= inp_channels,
            num_classes=2))
    elif configs['method'] == 'transunet_cd':
        models.append(TransUNet_CD(
            img_dim=configs['datasets']['img_size'],
            in_channels=inp_channels,
            out_channels=model_configs['out_channels'],
            head_num=model_configs['head_num'],
            mlp_dim=model_configs['mlp_dim'],
            block_num=model_configs['block_num'],
            patch_dim=model_configs['patch_dim'],
            class_num=2,
            siamese=model_configs['siamese']))

    # Put models to device
    models = [model.module.to(device) if isinstance(model, nn.DataParallel) else model.to(device) for model in models]

    # Load any checkpoints
    if checkpoint is not None:
        models[0].load_state_dict(checkpoint['model_state_dict'])

    return models


def compute_class_weights(configs):
    '''
    Computes the number of pixels per class (burnt/unburnt), then computes the weights of each class
    based on these counts and returns the weights.
    '''
    if configs['train']['weighted_loss']:
        img_size = configs['datasets']['img_size']

        burnt = {'train': 0, 'val': 0, 'test': 0}
        unburnt = {'train': 0, 'val': 0, 'test': 0}

        for mode in ['train', 'val', 'test']:
            pickle_file = pickle.load(open(Path(configs['paths']['dataset']) / f'{configs["dataset_type"]}' / configs['datasets'][mode], 'rb'))

            for k, v in pickle_file.items():
                if v['positive_flag']:
                    img = torch.load(v['label'])
                    counts = dict(zip(*[i.detach().cpu().tolist() for i in img.unique(sorted=True, return_counts=True)]))
                    if 1 in counts.keys(): burnt[mode] += counts[1]
                    if 0 in counts.keys(): unburnt[mode] += counts[0]
                else:
                    unburnt[mode] += (img_size * img_size)

        return {
            'train': ((burnt['train'] + unburnt['train']) / (2 * unburnt['train']), (burnt['train'] + unburnt['train']) / (2 * burnt['train'])),
            'val': ((burnt['val'] + unburnt['val']) / (2 * unburnt['val']), (burnt['val'] + unburnt['val']) / (2 * burnt['val'])),
            'test': ((burnt['test'] + unburnt['test']) / (2 * unburnt['test']), (burnt['test'] + unburnt['test']) / (2 * burnt['test']))
        }
    else:
        return {
            'train': (1, 1),
            'val': (1, 1),
            'test': (1, 1)
        }


class MyConfusionMatrix():
    def __init__(self, num_classes, ignore_index=None, device=None):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.device = device

        if ignore_index is not None:
            self.tp = torch.zeros(num_classes-1).to(torch.long).to(self.device)
            self.fp = torch.zeros(num_classes-1).to(torch.long).to(self.device)
            self.tn = torch.zeros(num_classes-1).to(torch.long).to(self.device)
            self.fn = torch.zeros(num_classes-1).to(torch.long).to(self.device)
        else:
            self.tp = torch.zeros(num_classes).to(torch.long).to(self.device)
            self.fp = torch.zeros(num_classes).to(torch.long).to(self.device)
            self.tn = torch.zeros(num_classes).to(torch.long).to(self.device)
            self.fn = torch.zeros(num_classes).to(torch.long).to(self.device)

        self.cm = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(self.device)
        self._accuracy = 0.
        self._precision = 0.
        self._recall = 0.
        self._f1_score = 0

    def compute(self, preds, target):
        new_cm = self.cm(preds, target)

        if self.ignore_index is not None:
            # Drop row and column corresponding to ignored index
            idx = list(set(range(new_cm.shape[0])) - set([self.ignore_index]))
            new_cm = new_cm[idx, :][:, idx]

        tp = torch.diag(new_cm)
        fp = torch.sum(new_cm, dim=0) - torch.diag(new_cm)
        fn = torch.sum(new_cm, dim=1) - torch.diag(new_cm)

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += torch.sum(new_cm) - (fp + fn + tp)

        self._accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        self._precision = self.tp / (self.tp + self.fp)
        self._recall = self.tp / (self.tp + self.fn)
        self._f1_score = (2 * self._precision * self._recall) / (self._precision + self._recall)

        return new_cm

    def accuracy(self):
        return self._accuracy

    def precision(self):
        return self._precision

    def recall(self):
        return self._recall

    def f1_score(self):
        return self._f1_score

    def reset(self):
        if self.ignore_index is not None:
            self.tp = torch.zeros(self.num_classes-1).to(torch.long)
            self.fp = torch.zeros(self.num_classes-1).to(torch.long)
            self.tn = torch.zeros(self.num_classes-1).to(torch.long)
            self.fn = torch.zeros(self.num_classes-1).to(torch.long)
        else:
            self.tp = torch.zeros(self.num_classes).to(torch.long)
            self.fp = torch.zeros(self.num_classes).to(torch.long)
            self.tn = torch.zeros(self.num_classes).to(torch.long)
            self.fn = torch.zeros(self.num_classes).to(torch.long)

        self.cm = ConfusionMatrix(num_classes=self.num_classes).to(self.device)
        self._accuracy = 0.
        self._precision = 0.
        self._recall = 0.
        self._f1_score = 0


class LandCoverMetrics():
    '''
    Logs metrics by land cover.

    Note: Only 0 and 1 classes are taken into account.
    '''
    def __init__(self, device=None):
        self.device = device

        self._lc_stats = {}
        self._lc_names = {
            0: '0 - NODATA',
            48: '999 - NODATA',
            128: '128 - NODATA',
            1: '111 - Continuous urban fabric',
            2: '112 - Discontinuous urban fabric',
            3: '121 - Industrial or commercial units',
            4: '122 - Road and rail networks and associated land',
            5: '123 - Port areas',
            6: '124 - Airports',
            7: '131 - Mineral extraction sites',
            8: '132 - Dump sites',
            9: '133 - Construction sites',
            10: '141 - Green urban areas',
            11: '142 - Sport and leisure facilities',
            12: '211 - Non-irrigated arable land',
            13: '212 - Permanently irrigated land',
            14: '213 - Rice fields',
            15: '221 - Vineyards',
            16: '222 - Fruit trees and berry plantations',
            17: '223 - Olive groves',
            18: '231 - Pastures',
            19: '241 - Annual crops associated with permanent crops',
            20: '242 - Complex cultivation patterns',
            21: '243 - Land principally occupied by agriculture with significant areas of natural vegetation',
            22: '244 - Agro-forestry areas',
            23: '311 - Broad-leaved forest',
            24: '312 - Coniferous forest',
            25: '313 - Mixed forest',
            26: '321 - Natural grasslands',
            27: '322 - Moors and heathland',
            28: '323 - Sclerophyllous vegetation',
            29: '324 - Transitional woodland-shrub',
            30: '331 - Beaches dunes sands',
            31: '332 - Bare rocks',
            32: '333 - Sparsely vegetated areas',
            33: '334 - Burnt areas',
            34: '335 - Glaciers and perpetual snow',
            35: '411 - Inland marshes',
            36: '412 - Peat bogs',
            37: '421 - Salt marshes',
            38: '422 - Salines',
            39: '423 - Intertidal flats',
            40: '511 - Water courses',
            41: '512 - Water bodies',
            42: '521 - Coastal lagoons',
            43: '522 - Estuaries',
            44: '523 - Sea and ocean'
        }

    def compute(self, preds, target, lc_map):
        land_covers = [i.item() for i in torch.unique(lc_map)]

        for lc in land_covers:
            if lc not in self._lc_stats.keys():
                self._lc_stats[lc] = {'tn_0': 0, 'tn_1': 0, 'fn_0': 0, 'fn_1': 0, 'tp_0': 0, 'tp_1': 0, 'fp_0': 0, 'fp_1': 0}
                self._lc_stats[lc]['iou'] = JaccardIndex(task='multiclass', num_classes=3, average='none', ignore_index=2).to(self.device)

            idx = torch.where(lc_map == lc)
            self._lc_stats[lc]['tp_0'] += ((preds[idx] == 0) & (target[idx] == 0)).sum()
            self._lc_stats[lc]['fp_0'] += ((preds[idx] == 0) & (target[idx] == 1)).sum()
            self._lc_stats[lc]['tn_0'] += ((preds[idx] == 1) & (target[idx] == 1)).sum()
            self._lc_stats[lc]['fn_0'] += ((preds[idx] == 1) & (target[idx] == 0)).sum()

            self._lc_stats[lc]['tp_1'] += ((preds[idx] == 1) & (target[idx] == 1)).sum()
            self._lc_stats[lc]['fp_1'] += ((preds[idx] == 1) & (target[idx] == 0)).sum()
            self._lc_stats[lc]['tn_1'] += ((preds[idx] == 0) & (target[idx] == 0)).sum()
            self._lc_stats[lc]['fn_1'] += ((preds[idx] == 0) & (target[idx] == 1)).sum()

            self._lc_stats[lc]['iou'].update(preds[idx], target[idx])

    def get_metrics(self):
        stats = {}
        for lc_id, lc_info in self._lc_stats.items():
            # if (lc_info['tn'] > 0) and (lc_info['fp'] == 0) and (lc_info['tp'] == 0) and (lc_info['fn'] == 0): continue

            stats[lc_id] = {'name': self._lc_names[lc_id]}

            stats[lc_id]['acc'] = (lc_info['tp_0'] + lc_info['tn_0']) / (lc_info['tp_0'] + lc_info['fp_0'] + lc_info['fn_0'] + lc_info['tn_0'])
            stats[lc_id]['prec'] = [
                lc_info['tp_0'] / (lc_info['tp_0'] + lc_info['fp_0']),
                lc_info['tp_1'] / (lc_info['tp_1'] + lc_info['fp_1'])
            ]
            stats[lc_id]['rec'] = [
                lc_info['tp_0'] / (lc_info['tp_0'] + lc_info['fn_0']),
                lc_info['tp_1'] / (lc_info['tp_1'] + lc_info['fn_1'])
            ]
            stats[lc_id]['f1'] = [
                (2 * stats[lc_id]['prec'][0] * stats[lc_id]['rec'][0]) / (stats[lc_id]['prec'][0] + stats[lc_id]['rec'][0]),
                (2 * stats[lc_id]['prec'][1] * stats[lc_id]['rec'][1]) / (stats[lc_id]['prec'][1] + stats[lc_id]['rec'][1])
            ]
            stats[lc_id]['iou'] = lc_info['iou'].compute()[:2]

            for metric in ['acc', 'prec', 'rec', 'f1', 'iou']:
                if isinstance(stats[lc_id][metric], list) or (stats[lc_id][metric].ndim > 0):
                    for i, mi in enumerate(stats[lc_id][metric]):
                        if torch.isnan(mi):
                            stats[lc_id][metric][i] = 0.0
                        else:
                            stats[lc_id][metric][i] = stats[lc_id][metric][i].item() * 100
                else:
                    if torch.isnan(stats[lc_id][metric]):
                        stats[lc_id][metric] = 0.0
                    else:
                        stats[lc_id][metric] = stats[lc_id][metric].item() * 100

        return stats

    def reset(self):
        self._lc_stats = {}


def initialize_metrics(configs, device):
    '''
    Initializes various metrics.
    '''
    if device is None:
        device = 'cpu'
    else:
        device = device

    cm = MyConfusionMatrix(num_classes=3, ignore_index=2, device=device)
    iou = JaccardIndex(task='multiclass', num_classes=3, average='none', ignore_index=2).to(device)

    return cm, iou


def create_loss(configs, mode, device, class_weights, model_configs=None):
    '''
    Initializes the loss function.
    '''
    if configs['train']['loss_function'] == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=torch.Tensor(class_weights[mode]), ignore_index=2).to(device)
    elif configs['train']['loss_function'] == 'focal':
        return torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.Tensor(class_weights[mode]),
            gamma=2,
            reduction='mean',
            force_reload=False,
            ignore_index=2).to(device)
    elif configs['train']['loss_function'] == 'dice':
        if ('activation' in model_configs.keys()) and (model_configs['activation'] is not None):
            use_softmax = False
        else:
            use_softmax = True
        return DiceLoss(ignore_index=2, use_softmax=use_softmax).to(device)
    elif configs['train']['loss_function'] == 'dice+ce':
        if ('activation' in model_configs.keys()) and (model_configs['activation'] is not None):
            use_softmax = False
        else:
            use_softmax = True
        return BCEandDiceLoss(weights=torch.Tensor(class_weights[mode]), ignore_index=2, use_softmax=use_softmax).to(device)
    else:
        raise NotImplementedError(f'Loss {configs["train"]["loss_function"]} is not implemented!')


def init_optimizer(model, checkpoint, configs, model_configs, model_name=None):
    '''
    Initialize the optimizer.
    '''
    if model_name is None:
        lr = model_configs['optimizer']['learning_rate']
        optim_args = model_configs['optimizer']
    else:
        lr = model_configs['optimizer'][model_name]['learning_rate']
        optim_args = model_configs['optimizer'][model_name]

    if optim_args['name'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=optim_args['weight_decay'])
    elif optim_args['name'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=optim_args['weight_decay'])
    elif optim_args['name'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=optim_args['weight_decay'], momentum=optim_args['momentum'])
    else:
        raise NotImplementedError(f'Optimizer {optim_args["name"]} is not implemented!')

    # Load checkpoint (if any)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def init_lr_scheduler(optimizer, checkpoint, configs, model_configs, model_name=None):
    # Get the required LR scheduling
    if model_name is not None:
        lr_schedule = model_configs['optimizer'][model_name]['lr_schedule']
        optim_args = model_configs['optimizer'][model_name]
    else:
        lr_schedule = model_configs['optimizer']['lr_schedule']
        optim_args = model_configs['optimizer']

    # Initialize the LR scheduler
    if lr_schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, optim_args['lr_schedule_steps'])
    elif lr_schedule is None:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1, last_epoch=-1)
    elif lr_schedule == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(configs['train']['n_epochs'] + 1)
            return lr_l
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_schedule == 'step':
        if 'lr_scheduler_gamma' in optim_args.keys():
            gamma = optim_args['lr_scheduler_gamma'] = 0.5
        else:
            gamma = 0.1

        if 'lr_scheduler_step' in optim_args.keys():
            step_size = optim_args['lr_scheduler_step']
        else:
            step_size = configs['train']['n_epochs'] // 3

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_schedule.startswith('step_'):
        step_size = int(lr_schedule.split('_')[1])

        if 'lr_scheduler_gamma' in optim_args.keys():
            gamma = optim_args['lr_scheduler_gamma'] = 0.5
        else:
            gamma = 0.1

        if 'lr_scheduler_step' in optim_args.keys():
            step_size = optim_args['lr_scheduler_step']
        else:
            step_size = configs['train']['n_epochs'] // 3

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise NotImplementedError(f'{lr_schedule} LR scheduling is not yet implemented!')


    # Load checkpoint (if any)
    if checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    return lr_scheduler


def get_sample_index_in_batch(batch_size, idx):
    '''
    Takes as input the index of an individual sample (as it is mapped by the Dataset object) and
    calculates the batch index it is contained into, as well as its index inside the batch.
    '''
    batch_idx = (idx // batch_size)
    idx_in_batch = (idx % batch_size)

    return batch_idx, idx_in_batch
