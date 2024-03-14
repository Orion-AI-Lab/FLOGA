from tqdm import tqdm
import wandb
import pyjson5
import numpy as np

import torch

from utils import (
    initialize_metrics,
    create_loss,
    init_optimizer,
    init_lr_scheduler,
    get_sample_index_in_batch,
    LandCoverMetrics
)


CLASS_LABELS = {0: 'Unburnt', 1: 'Burnt', 2: 'Other events'}


def train_change_detection(model, device, class_weights, run_path, init_epoch, train_loader, val_loader, validation_id,
                           gsd, checkpoint, configs, model_configs, rep_i, wandb=None):
    '''
    Train a model for Change Detection using a single satellite source.
    '''
    print(f'\n===== REP {rep_i} =====\n')

    (run_path / 'checkpoints' / f'{rep_i}').mkdir(parents=True, exist_ok=True)

    # Initialize metrics
    cm, iou = initialize_metrics(configs, device)

    if isinstance(configs['train']['save_checkpoint_freq'], list):
        save_every, save_last_epoch = [int(i) for  i in configs['train']['save_checkpoint_freq']]
    else:
        save_every = configs['train']['save_checkpoint_freq']
        save_last_epoch = 0

    # Initialize loss function
    criterion = create_loss(configs, 'train', device, class_weights, model_configs=model_configs)

    # Initialize optimizer
    optimizer = init_optimizer(model, checkpoint, configs, model_configs)

    # Initialize LR scheduling
    lr_scheduler = init_lr_scheduler(optimizer, checkpoint, configs, model_configs)

    if configs['paths']['load_state'] is not None:
        # Get the best validation f-score up to now
        with open(run_path / 'checkpoints' / f'{rep_i}' / 'best_segmentation.txt', 'r') as f:
            best_val = float(f.readlines()[-1].strip())
    else:
        best_val = 0.0
    best_stats = {}

    if configs['train']['mixed_precision']:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()

    total_iters = 0
    last_epoch = init_epoch + configs['train']['n_epochs'] + 1

    bands_idx = list(configs['datasets']['selected_bands'][configs['datasets']['data_source']].values())
    inds = None

    for epoch in range(init_epoch, last_epoch):
        model.train()

        train_loss = 0.0

        with tqdm(initial=0, total=len(train_loader)) as pbar:
            for index, batch in enumerate(train_loader):
                if configs['datasets']['data_source'] == 'mod':
                    before_img = batch['MOD_before_image'][:, bands_idx, :, :]
                    after_img = batch['MOD_after_image'][:, bands_idx, :, :]
                else:
                    before_img = batch['S2_before_image'][:, bands_idx, :, :]
                    after_img = batch['S2_after_image'][:, bands_idx, :, :]
                label = batch['label']

                with torch.cuda.amp.autocast(enabled=configs['train']['mixed_precision']):
                    before_img = before_img.to(device)
                    after_img = after_img.to(device)
                    label = label.to(device).long()

                    optimizer.zero_grad()

                    output = model(before_img, after_img)

                    if configs['method'] == 'changeformer':
                        if model_configs['multi_scale_infer']:
                            final_output = torch.zeros(output[-1].size()).to(configs['device'])
                            for pred in output:
                                if pred.size(2) != output[-1].size(2):
                                    final_output = final_output + F.interpolate(pred, size=output[-1].size(2), mode="nearest")
                                else:
                                    final_output = final_output + pred
                            final_output = final_output / len(output)
                        else:
                            final_output = output[-1]

                        predictions = final_output.argmax(1).to(dtype=torch.int8)
                    else:
                        predictions = output.argmax(1).to(dtype=torch.int8)

                    if configs['method'] == 'changeformer':
                        if model_configs['multi_scale_train']:
                            i = 0
                            temp_loss = 0.0
                            for pred in output:
                                if pred.size(2) != label.size(2):
                                    temp_loss = temp_loss + model_configs['multi_pred_weights'][i] * criterion(pred, F.interpolate(label, size=pred.size(2), mode="nearest"))
                                else:
                                    temp_loss = temp_loss + model_configs['multi_pred_weights'][i] * criterion(pred, label)
                                i+=1
                            loss = temp_loss
                        else:
                            loss = criterion(output[-1], label)
                    else:
                        loss = criterion(output, label)

                    # Note: loss.item() is averaged across all training examples of the current batch
                    # so we multiply by the batch size to obtain the unaveraged current loss
                    train_loss += (loss.item() * before_img.size(0))

                if configs['train']['mixed_precision']:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                cm.compute(predictions, label)

                acc = cm.accuracy()
                iou.update(predictions, label)

                if index % configs['train']['print_freq'] == 0:
                    pbar.set_description(f'({epoch}) Train Loss: {train_loss:.4f}')

                pbar.update(1)

        acc = cm.accuracy()
        score = cm.f1_score()
        prec = cm.precision()
        rec = cm.recall()
        ious = iou.compute()
        mean_iou = ious[:2].mean()

        print(f'F1-score: {score[1].item()}')

        # Calculate average loss over an epoch
        train_loss = train_loss / len(train_loader)

        lrs = lr_scheduler.get_last_lr()[0]

        if (save_every != -1) and ((epoch >= save_last_epoch) or (epoch % save_every == 0)):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss
                }, run_path / 'checkpoints' / f'{rep_i}' / f'checkpoint_epoch={epoch}.pt')

        if configs['wandb']['activate']:
            loss_val = loss.item()

            log_dict = {
                f'({rep_i}) Epoch': epoch,
                f'({rep_i}) Iteration': index,
                f'({rep_i}) Train Loss': loss_val,
                f'({rep_i}) Train Accuracy ({CLASS_LABELS[0]})': 100 * acc[0].item(),
                f'({rep_i}) Train Accuracy ({CLASS_LABELS[1]})': 100 * acc[1].item(),
                f'({rep_i}) Train F-Score ({CLASS_LABELS[0]})': 100 * score[0].item(),
                f'({rep_i}) Train F-Score ({CLASS_LABELS[1]})': 100 * score[1].item(),
                f'({rep_i}) Train Precision ({CLASS_LABELS[0]})': 100 * prec[0].item(),
                f'({rep_i}) Train Precision ({CLASS_LABELS[1]})': 100 * prec[1].item(),
                f'({rep_i}) Train Recall ({CLASS_LABELS[0]})': 100 * rec[0].item(),
                f'({rep_i}) Train Recall ({CLASS_LABELS[1]})': 100 * rec[1].item(),
                f'({rep_i}) Train IoU ({CLASS_LABELS[0]})': 100 * ious[0].item(),
                f'({rep_i}) Train IoU ({CLASS_LABELS[1]})': 100 * ious[1].item(),
                f'({rep_i}) Train MeanIoU': mean_iou * 100,
                f'({rep_i}) lr': lr_scheduler.get_last_lr()[0]
            }

            wandb.log(log_dict)

        # Update LR scheduler
        lr_scheduler.step()

        # Save the current learning rate
        lrs = [lr_scheduler.get_last_lr()[0]]

        with open(run_path / f'{rep_i} lrs.txt', 'a') as f:
            f.write(f'{epoch}: {", ".join([str(lr) for lr in lrs])}\n')

        # Evaluate on validation set
        val_acc, val_score, miou, burnt_score = eval_change_detection(model, device, class_weights,
            init_epoch, val_loader, validation_id, gsd, 'Val', configs, model_configs, rep_i, wandb, run_path)

        if (epoch != 0) and (burnt_score > best_val):
            best_val = burnt_score
            best_stats['acc'] = best_val
            best_stats['epoch'] = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss
                }, run_path / 'checkpoints' / f'{rep_i}' / 'best_segmentation.pt')

            with open(run_path / 'checkpoints' / f'{rep_i}' / 'best_segmentation.txt', 'w') as f:
                f.write(f'{epoch}\n')
                f.write(f'{burnt_score}')


def eval_change_detection(model, device, class_weights, init_epoch, loader, validation_id, gsd,
                          mode, configs, model_configs, rep_i, wandb=None, run_path=None):
    cm, iou = initialize_metrics(configs, device)

    if configs['train']['log_landcover_metrics']:
        lc_logger = LandCoverMetrics(device)

    criterion = create_loss(configs, 'val', device, class_weights, model_configs=model_configs)

    total_loss = 0.0
    total_iters = 0

    bands_idx = list(configs['datasets']['selected_bands'][configs['datasets']['data_source']].values())

    batch_idx, idx_in_batch = get_sample_index_in_batch(configs['datasets']['batch_size'], validation_id)

    inds = None

    model.eval()

    with tqdm(initial=0, total=len(loader)) as pbar:
        for index, batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=configs['train']['mixed_precision']):
                with torch.no_grad():
                    if configs['datasets']['data_source'] == 'mod':
                        before_img = batch['MOD_before_image'][:, bands_idx, :, :]
                        after_img = batch['MOD_after_image'][:, bands_idx, :, :]
                    else:
                        before_img = batch['S2_before_image'][:, bands_idx, :, :]
                        after_img = batch['S2_after_image'][:, bands_idx, :, :]
                    label = batch['label']

                    before_img = before_img.to(device)
                    after_img = after_img.to(device)
                    label = label.to(device).long()

                    output = model(before_img, after_img)

                    if configs['method'] == 'changeformer':
                        output = output[-1]

                    predictions = output.argmax(1).to(dtype=torch.int8)

                    loss = criterion(output, label)

                    # Note: loss.item() is averaged across all training examples of the current batch
                    # so we multiply by the batch size to obtain the unaveraged current loss
                    total_loss += (loss.item() * before_img.size(0))

                    cm.compute(predictions, label)
                    iou.update(predictions, label)

                    if configs['train']['log_landcover_metrics']:
                        clc = batch['clc_mask'].to(device)
                        lc_logger.compute(predictions, label, clc)

                    if index % configs['train']['print_freq'] == 0:
                        pbar.set_description(f'{mode} Loss: {total_loss:.4f}')

                    if configs['wandb']['activate'] and (index == batch_idx):
                        # Note: permute() is used because wandb Image requires channel-last format
                        before_img_wand = before_img[idx_in_batch].permute(1, 2, 0).detach().cpu()
                        after_img_wand = after_img[idx_in_batch].permute(1, 2, 0).detach().cpu()

                        label_wand = label[idx_in_batch].detach().cpu()
                        prediction_wand = predictions[idx_in_batch].detach().cpu()

            pbar.update(1)

    acc = cm.accuracy()
    score = cm.f1_score()
    prec = cm.precision()
    rec = cm.recall()
    ious = iou.compute()
    mean_iou = ious[:2].mean()

    if configs['train']['log_landcover_metrics']:
        lc_stats = lc_logger.get_metrics()

    print(f'VAL F1-score: {score[1].item()}')

    selected_bands_idx = {band: order_id for order_id, (band, _) in enumerate(configs['datasets']['selected_bands'][configs['datasets']['data_source']].items())}

    if configs['datasets']['data_source'] == 'sen2':
        if gsd['sen2'] == '10':
            if set(['B08', 'B04', 'B03']) <= set(configs['datasets']['selected_bands']['sen2'].keys()):
                # NIR, Red, Green
                bands_to_plot = [selected_bands_idx[band] for band in ['B08', 'B04', 'B03']]
            else:
                # Plot the first band
                bands_to_plot = list(selected_bands_idx.values())[0]
        else:
            if set(['B8A', 'B04', 'B03']) <= set(configs['datasets']['selected_bands']['sen2'].keys()):
                # NIR, Red, Green
                bands_to_plot = [selected_bands_idx[band] for band in ['B8A', 'B04', 'B03']]
            else:
                # Plot the first band
                bands_to_plot = list(selected_bands_idx.values())[0]
    elif configs['datasets']['data_source'] == 'mod':
        if set(['B02', 'B01', 'B04']) <= configs['datasets']['selected_bands']['mod'].keys():
            # NIR, Red, Green
            bands_to_plot = [selected_bands_idx[band] for band in ['B02', 'B01', 'B04']]
        else:
            # Plot the first band
            bands_to_plot = list(selected_bands_idx.values())[0]

    if configs['wandb']['activate']:
        if len(bands_to_plot) == 3:
            before_img_log = wandb.Image(
                (before_img_wand[:, :, bands_to_plot] * 255).int().numpy(),
                caption='Before',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            after_img_log = wandb.Image(
                (after_img_wand[:, :, bands_to_plot] * 255).int().numpy(),
                caption='After',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            wandb.log({f'({rep_i}) {mode} Before image': before_img_log})
            wandb.log({f'({rep_i}) {mode} After image': after_img_log})
        else:
            before_img_red_log = wandb.Image(
                (before_img_wand[:, :, bands_to_plot[1]] * 255).int().numpy(),
                caption='Before (Red)',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            before_img_nir_log = wandb.Image(
                (before_img_wand[:, :, bands_to_plot[0]] * 255).int().numpy(),
                caption='Before (NIR)',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            after_img_red_log = wandb.Image(
                (after_img_wand[:, :, bands_to_plot[1]] * 255).int().numpy(),
                caption='After (Red)',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            after_img_nir_log = wandb.Image(
                (after_img_wand[:, :, bands_to_plot[0]] * 255).int().numpy(),
                caption='After (NIR)',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            wandb.log({f'({rep_i}) {mode} Before image (Red)': before_img_red_log})
            wandb.log({f'({rep_i}) {mode} Before image (NIR)': before_img_nir_log})
            wandb.log({f'({rep_i}) {mode} After image (Red)': after_img_red_log})
            wandb.log({f'({rep_i}) {mode} After image (NIR)': after_img_nir_log})

    if configs['wandb']['activate']:
        wandb.log({
            f'({rep_i}) {mode} F-Score ({CLASS_LABELS[0]})': 100 * score[0].item(),
            f'({rep_i}) {mode} F-Score ({CLASS_LABELS[1]})': 100 * score[1].item(),
            f'({rep_i}) {mode} IoU ({CLASS_LABELS[0]})': 100 * ious[0],
            f'({rep_i}) {mode} IoU ({CLASS_LABELS[1]})': 100 * ious[1],
            f'({rep_i}) {mode} Precision ({CLASS_LABELS[0]})': 100 * prec[0].item(),
            f'({rep_i}) {mode} Precision ({CLASS_LABELS[1]})': 100 * prec[1].item(),
            f'({rep_i}) {mode} Recall ({CLASS_LABELS[0]})': 100 * rec[0].item(),
            f'({rep_i}) {mode} Recall ({CLASS_LABELS[1]})': 100 * rec[1].item(),
            f'({rep_i}) {mode} Accuracy ({CLASS_LABELS[0]})': 100 * acc[0].item(),
            f'({rep_i}) {mode} Accuracy ({CLASS_LABELS[1]})': 100 * acc[1].item(),
            f'({rep_i}) {mode} MeanIoU': 100 * mean_iou.item(),
            f'({rep_i}) {mode} Loss': total_loss / len(loader)
        })
    elif mode == 'test':
        print(f'({rep_i}) {mode} F-Score ({CLASS_LABELS[0]}): {100 * score[0].item()}')
        print(f'({rep_i}) {mode} F-Score ({CLASS_LABELS[1]}): {100 * score[1].item()}')
        print(f'({rep_i}) {mode} IoU ({CLASS_LABELS[0]}): {100 * ious[0]}')
        print(f'({rep_i}) {mode} IoU ({CLASS_LABELS[1]}): {100 * ious[1]}')
        print(f'({rep_i}) {mode} Precision ({CLASS_LABELS[0]}): {100 * prec[0].item()}')
        print(f'({rep_i}) {mode} Precision ({CLASS_LABELS[1]}): {100 * prec[1].item()}')
        print(f'({rep_i}) {mode} Recall ({CLASS_LABELS[0]}): {100 * rec[0].item()}')
        print(f'({rep_i}) {mode} Recall ({CLASS_LABELS[1]}): {100 * rec[1].item()}')
        print(f'({rep_i}) {mode} Accuracy ({CLASS_LABELS[0]}): {100 * acc[0].item()}')
        print(f'({rep_i}) {mode} Accuracy ({CLASS_LABELS[1]}): {100 * acc[1].item()}')
        print(f'({rep_i}) {mode} MeanIoU {100 * mean_iou.item()}')

        if configs['train']['log_landcover_metrics']:
            print('')
            for lc_id, lc_info in lc_stats.items():
                print(f'{lc_id}: {lc_info}')

    if mode == 'test':
        res = {
            'precision': (100 * prec[0].item(), 100 * prec[1].item()),
            'recall': (100 * rec[0].item(), 100 * rec[1].item()),
            'accuracy': (100 * acc[0].item(), 100 * acc[1].item()),
            'f1': (100 * score[0].item(), 100 * score[1].item()),
            'iou': (100 * ious[0].item(), 100 * ious[1].item())
        }

        if configs['train']['log_landcover_metrics']:
            res['lc_stats'] = lc_stats

        return res
    else:
        return 100 * acc.nanmean(), 100 * score.nanmean(), 100 * mean_iou, score[1]
