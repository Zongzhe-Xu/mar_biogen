import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import copy
import time


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        
        # Handle labels - text conditioning uses strings, class conditioning uses tensors
        if not isinstance(labels, (list, tuple)):
            # Class label conditioning - move to device
            labels = labels.to(device, non_blocking=True)
        # For text conditioning, labels are strings and will be handled in the model

        # forward - directly pass time series data to model
        with torch.cuda.amp.autocast():
            loss = model(samples, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True):
    model_without_ddp.eval()
    num_steps = args.num_samples // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}cfg{}-samples{}-epoch{}".format(args.num_iter,
                                                                                                     args.num_sampling_steps,
                                                                                                     args.temperature,
                                                                                                     args.cfg_schedule,
                                                                                                     cfg,
                                                                                                     args.num_samples,
                                                                                                     epoch))
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_sample_cnt = 0

    # Prepare labels for generation
    if hasattr(model_without_ddp, 'use_text_conditioning') and model_without_ddp.use_text_conditioning:
        # Text conditioning: use empty strings for now
        class_num = args.class_num
        assert args.num_samples % class_num == 0  # number of samples per class must be the same
        # Create empty text prompts for each class
        text_prompts = ["" for j in range(class_num)]
        text_prompts_world = text_prompts * (args.num_samples // class_num)
        text_prompts_world = text_prompts_world + [""] * 50000  # padding
    else:
        # Class label conditioning
        class_num = args.class_num
        assert args.num_samples % class_num == 0  # number of samples per class must be the same
        class_label_gen_world = np.arange(0, class_num).repeat(args.num_samples // class_num)
        class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        if hasattr(model_without_ddp, 'use_text_conditioning') and model_without_ddp.use_text_conditioning:
            # Text conditioning
            labels_gen = text_prompts_world[world_size * batch_size * i + local_rank * batch_size:
                                           world_size * batch_size * i + (local_rank + 1) * batch_size]
        else:
            # Class label conditioning
            labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                              world_size * batch_size * i + (local_rank + 1) * batch_size]
            labels_gen = torch.Tensor(labels_gen).long().to(next(model_without_ddp.parameters()).device)

        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                sampled_timeseries = model_without_ddp.sample_tokens(bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
                                                                   cfg_schedule=args.cfg_schedule, labels=labels_gen,
                                                                   temperature=args.temperature)

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_sample_cnt += batch_size
            print("Generating {} samples takes {:.5f} seconds, {:.5f} sec per sample".format(gen_sample_cnt, used_time, used_time / gen_sample_cnt))

        # Only call barrier if distributed is initialized
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        sampled_timeseries = sampled_timeseries.detach().cpu()

        # distributed save
        for b_id in range(sampled_timeseries.size(0)):
            sample_id = i * sampled_timeseries.size(0) * world_size + local_rank * sampled_timeseries.size(0) + b_id
            if sample_id >= args.num_samples:
                break
            # Save as numpy array
            sample_data = sampled_timeseries[b_id].numpy()
            np.save(os.path.join(save_folder, '{}.npy'.format(str(sample_id).zfill(5))), sample_data)

    # Only call barrier if distributed is initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # Log generation stats
    if log_writer is not None:
        postfix = ""
        if use_ema:
           postfix = postfix + "_ema"
        if not cfg == 1.0:
           postfix = postfix + "_cfg{}".format(cfg)
        log_writer.add_scalar('generation_time{}'.format(postfix), used_time / gen_sample_cnt, epoch)
        print("Generation time per sample: {:.4f} seconds".format(used_time / gen_sample_cnt))
        # remove temporal saving folder
        shutil.rmtree(save_folder)

    # Only call barrier if distributed is initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    time.sleep(10)
