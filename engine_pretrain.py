# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from test_npid import NN, kNN

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, data_loader_val: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    lemniscate=None,
                    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _, index) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss_mae, _, _, loss_npid, _= model(samples, mask_ratio=args.mask_ratio, index = index, is_train=True)
            loss = loss_mae + 0.1 * loss_npid

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()


        metric_logger.update(loss_all=loss_value)
        metric_logger.update(loss_mae=loss_mae.item())
        metric_logger.update(loss_npid=loss_npid.item())


        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('origin_loss/train_loss_mae', loss_mae.item(), epoch_1000x)
            log_writer.add_scalar('origin_loss/train_loss_npid', loss_npid.item(), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    pred1 = NN(epoch, model, lemniscate, data_loader, data_loader_val)
    log_writer.add_scalar('NN_ac', pred1, epoch)
    if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):

        top1 = kNN(0, model, lemniscate, data_loader, data_loader_val, 200, args.nce_t)
        log_writer.add_scalar('KNN_top1', top1, epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred1