"""Manage training for SMoEs."""

import argparse
import random
import functools
import time
import sys
import os
import os.path

import yaml
import numpy as np
import torch
from torch.nn import functional as F
import torch.backends.cudnn
import apex

from smoe.utils.distributed import (get_world_rank, get_world_size,
                                    get_local_size, get_local_rank,
                                    get_num_gpus, initialize_dist,
                                    get_cuda_device, get_job_id,)
from smoe.utils.tracker import (AverageTracker, AverageTrackerDevice)
from smoe.utils.metrics import MetricManager
from smoe.utils.log import (Logger, DataLogger)
from smoe.utils.misc import (ReduceLROnPlateauPatch, count_parameters, get_correct_indices_by_tol)
from smoe.data import get_dataloaders
from smoe.models import construct_model
import smoe.models.init
from smoe.models.loss import (MaskScaledLoss, routing_classification_loss,
                              routing_classification_loss_by_error)
import smoe.models.smoe_routing


def get_args():
    """Get arguments from the command line and config."""
    parser = argparse.ArgumentParser(description='SMoE training support')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file to load')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--job-id', type=str, default=None,
                        help='Job identifier')
    parser.add_argument('--print-freq', type=int, default=None,
                        help='Frequency for printing batch info')
    parser.add_argument('--no-print-stdout', default=False, action='store_true',
                        help='Do not print to stdout')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-eval', default=False, action='store_true',
                        help='Do not evaluate on validation set each epoch')
    parser.add_argument('--eval-on-init', default=False, action='store_true',
                        help='Run validation on network after initialization')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from given checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Epoch to resume from (loaded from checkpoint)')
    parser.add_argument('--save', type=str, default=[], nargs='+',
                        choices=['weights', 'grads', 'preds', 'routing'],
                        help='Save quantities during training')
    parser.add_argument('--save-on-best', default=False, action='store_true',
                        help='Only save quantities when there is a new best'
                             'validation score')

    # Data/training.
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data to train with')
    parser.add_argument('--dataset', type=str, default='heat',
                        choices=['heat'],
                        help='Dataset to use')
    parser.add_argument('--download', default=False, action='store_true',
                        help='Download dataset and exit')
    parser.add_argument('--data-context', type=int, default=0,
                        help='Number of additional samples for context')
    parser.add_argument('--data-maxmin-norm', default=False, action='store_true',
                        help='Use max-min normalization instead of standardization')
    parser.add_argument('--data-no-norm', default=False, action='store_true',
                        help='Do not normalize data')
    parser.add_argument('--mask', type=str, default=None,
                        help='Specify associated mask defining the domain')
    parser.add_argument('--data-mask', default=False, action='store_true',
                        help='Include the mask as an additional input channel')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs to train for')
    parser.add_argument('--drop-last', default=False, action='store_true',
                        help='Drop last small mini-batch')
    parser.add_argument('--data-shape', type=int, nargs='+', default=None,
                        help='Override input data dimensions')
    parser.add_argument('--data-target-shape', type=int, nargs='+', default=None,
                        help='Override target data dimensions')
    parser.add_argument('--data-mean', type=float, default=None,
                        help='Manually specify dataset mean')
    parser.add_argument('--data-std', type=float, default=None,
                        help='Manually specify dataset standard deviation')
    parser.add_argument('--metric', type=str, nargs='+', default=[],
                        choices=['all',
                                 'mse', 'mae', 'rmse', 'nrmse', 'prcntclose',
                                 'mask-mse', 'mask-mae', 'mask-rmse', 'mask-nrmse',
                                 'mask-prcntclose', 'top1', 'top5'],
                        help='Additional metrics to log')
    parser.add_argument('--prcntclose', type=float, nargs='+', default=[0.01],
                        help='Specify prcntclose metric tolerance')
    parser.add_argument('--prcntclose-tol-scale', type=float, nargs='+',
                        default=None,
                        help='Per-region tolerance scale')

    # Optimization.
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'huber', 'scaled-mse', 'scaled-huber', 'ce', 'bce'],
                        help='Loss criterion')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Delta value for Huber loss')
    parser.add_argument('--loss-region-scales', type=float, nargs='+', default=None,
                        help='Per-region loss scale')
    parser.add_argument('--rc-loss', default=False, action='store_true',
                        help='Train SMoE gates using routing classification')
    parser.add_argument('--rc-loss-from-loss', default=False, action='store_true',
                        help='Compute routing classification loss from real loss')
    parser.add_argument('--rc-loss-quantile', type=float, default=0.3,
                        help='Quantile for determining routing classification mistakes')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer algorithm')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--early-stop', type=int, default=None,
                        help='Stop if no improvement after this many epochs')
    parser.add_argument('--stop-on-metric-level', type=float, default=None,
                        help='Stop when opt metric reaches this value')
    parser.add_argument('--opt-metric', type=str, default='loss',
                        help='Metric to use for early stopping, plateauing, etc.')
    parser.add_argument('--opt-metric-max', default=False, action='store_true',
                        help='Maximize opt-metric rather than minimize')
    parser.add_argument('--schedule', type=str, default=None,
                        choices=['none', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--plateau-epochs', type=int, default=None,
                        help='Patience before dropping learning rate')
    parser.add_argument('--plateau-factor', type=float, default=0.1,
                        help='Amount to drop learning rate by on plateau')
    parser.add_argument('--loss-scale', type=float, default=None,
                        help='Scale loss by this value')
    parser.add_argument('--clip-grad', type=float, default=None,
                        help='Clip gradients to this range')
    parser.add_argument('--importance-loss-weight', type=float, default=0.0,
                        help='Weight for SMoE expert importance loss (0 disables)')
    parser.add_argument('--load-loss-weight', type=float, default=0.0,
                        help='Weight for SMoE expert load loss (0 disables)')
    parser.add_argument('--spatial-agreement-loss-weight', type=float, default=0.0,
                        help='Weight for SMoE spatial agreement loss (0 disables)')

    # Model specification.
    parser.add_argument('--model', type=str, default='conv',
                        help='Model to train')
    parser.add_argument('--initialization', type=str, default='default',
                        choices=['default', 'pytorch'],
                        help='How to initialize layers')
    parser.add_argument('--act', type=str, default='ReLU',
                        help='Activation function to use')
    parser.add_argument('--residuals', default=False, action='store_true',
                        help='Use residual connections')
    parser.add_argument('--no-batchnorm', default=False, action='store_true',
                        help='Do not use batchnorm')
    parser.add_argument('--conv-filters', type=int, nargs='+', default=[],
                        help='Number of filters in each conv layer')
    parser.add_argument('--num-neurons', type=int, nargs='+', default=[],
                        help='Number of neurons in each MLP')
    parser.add_argument('--layers', type=str, nargs='+', default=[],
                        help='Specify all layer types')
    parser.add_argument('--with-bias', default=False, action='store_true',
                        help='Use channel-wise bias on layers')
    parser.add_argument('--with-height-bias', default=False, action='store_true',
                        help='Use height-wise bias on layers')
    parser.add_argument('--with-width-bias', default=False, action='store_true',
                        help='Use width-wise bias on layers')
    parser.add_argument('--num-experts', type=int, nargs='+', default=[],
                        help='Number of experts in each layer')
    parser.add_argument('--last-layer-experts', type=int, default=1,
                        help='Number of experts in last layer')
    parser.add_argument('--classification-layer', default=False, action='store_true',
                        help='Classification MLP for final layer')
    parser.add_argument('--classification-gap', default=False, action='store_true',
                        help='Classification layer uses global average pooling')
    parser.add_argument('--expert-kernel-size', type=int, default=3,
                        help='Kernel size for experts (if used)')
    parser.add_argument('--gate-kernel-size', type=int, default=3,
                        help='Kernel size for gates (if used)')
    parser.add_argument('--unweighted-smoe', default=False, action='store_true',
                        help='Do not weight SMoE experts with gate output')
    parser.add_argument('--norm-weighted', default=False, action='store_true',
                        help='Use softmax weighting in SMoE')
    parser.add_argument('--absval-routing', default=False, action='store_true',
                        help='Use absolute value in SMoE')
    parser.add_argument('--noise', default=False, action='store_true',
                        help='Add noise to SMoE gating')
    parser.add_argument('--smoe-expert-block', type=str, default='conv',
                        help='Layer type for SMoE expert')
    parser.add_argument('--gate-type', type=str, default='conv',
                        help='SMoE gate type')
    parser.add_argument('--gate-add-mask', default=False, action='store_true',
                        help='Have each gate concatenate the region mask to its input')
    parser.add_argument('--gate-orthogonal', default=False, action='store_true',
                        help='Orthogonally init gate weights')
    parser.add_argument('--gate-act', type=str, default='id',
                        help='Action function after SMoE gate')
    parser.add_argument('--smooth-gate-error', default=False,
                        action='store_true',
                        help='Smooth the error signal to the SMoE gate')
    parser.add_argument('--dampen-expert-error', default=False,
                        action='store_true',
                        help='Dampen the error signal to the SMoE experts')
    parser.add_argument('--dampen-expert-error-factor', type=float, default=0.0,
                        help='Factor to dampen expert error signals by')
    parser.add_argument('--routing-error-quantile', type=float, default=0.7,
                        help='Quantile used to determine incorrect routings')

    # Performance.
    parser.add_argument('--dist', default=False, action='store_true',
                        help='Do distributed training')
    parser.add_argument('--rendezvous', type=str, default='file',
                        help='Distributed initialization scheme (file, tcp)')
    parser.add_argument('--fp16', default=False, action='store_true',
                        help='Use FP16/AMP training')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers for reading samples')
    parser.add_argument('--no-cudnn-bm', default=False, action='store_true',
                        help='Do not do benchmarking to select cuDNN algorithms')

    # Debugging.
    parser.add_argument('--dump-backprop-graph', default=False,
                        action='store_true',
                        help='Dump backprop graph then exit')
    parser.add_argument('--check-grads', default=False, action='store_true',
                        help='Check parameter gradients for bad values')
    parser.add_argument('--hang-on-bad-grad', default=False,
                        action='store_true',
                        help='Trigger a breakpoint if a bad grad is found')
    parser.add_argument('--deterministic', default=False, action='store_true',
                        help='Force PyTorch to be deterministic')

    # For simplifying batch runs where we don't need distributed training.
    parser.add_argument('--set-gpu-rank', default=False, action='store_true',
                        help='Select GPU device based on rank')
    parser.add_argument('--per-rank-config', default=None, type=str, nargs='+',
                        help='Specify a separate config file for each rank')

    args = parser.parse_args()

    # Load config file if present. Overrides CLI arguments.
    args_dict = vars(args)
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.full_load(f)
        args_dict.update(config)
    if args.per_rank_config:
        if len(args.per_rank_config) != get_world_size(required=True):
            raise RuntimeError('Not enough per-rank configs specified')
        # Load only the config corresponding to our rank.
        with open(args.per_rank_config[get_world_rank(required=True)]) as f:
            config = yaml.full_load(f)
        args_dict.update(config)

    return args


def aggregate_aux_losses(net):
    """Gather and sum all auxiliary losses recorded in net.

    This looks for a member named 'aux_losses' in each module. If
    present, it should be a dict whose values are the auxiliary losses
    to aggregate.

    All auxiliary losses will be summed together, and then added to the
    main loss.

    """
    aux_loss = None
    for module in net.modules():
        if hasattr(module, 'aux_losses'):
            for loss in getattr(module, 'aux_losses').values():
                if aux_loss is None:
                    aux_loss = loss
                elif loss is not None:
                    aux_loss = aux_loss + loss
    return aux_loss


def train(args, train_loader, net, scaler, criterion, optimizer,
          log, data_logger):
    """Perform one epoch of training."""
    net.train()
    losses = AverageTrackerDevice(len(train_loader), get_cuda_device(),
                                  allreduce=args.dist)
    batch_times = AverageTracker()
    end_time = time.perf_counter()
    for batch, data in enumerate(train_loader):
        samples, targets = data
        samples = samples.to(get_cuda_device(), non_blocking=True)
        targets = targets.to(get_cuda_device(), non_blocking=True)

        loss_vals = []
        with torch.cuda.amp.autocast(enabled=args.fp16):  # pyright: ignore reportPrivateImportusage
            output = net(samples)
            loss = criterion(output, targets)
            aux_loss = aggregate_aux_losses(net)
            if aux_loss is not None:
                loss += aux_loss
            if args.loss_scale:
                loss *= args.loss_scale
            loss_vals.append(loss)
            if args.rc_loss and args.rc_loss_from_loss:
                loss_vals += routing_classification_loss(net, output, targets)

        # Optionally visualize the backprop graph and then exit.
        if args.dump_backprop_graph:
            import torchviz
            dot = torchviz.make_dot(loss, params=dict(net.named_parameters()),
                                    show_attrs=True, show_saved=True)
            dot.render()
            sys.exit(0)
        losses.update(loss, samples.size(0))
        # We need to perform the regular backprop first to get the error
        # signals for the RC loss.
        if args.rc_loss and not args.rc_loss_from_loss:
            for loss_val in loss_vals:
                scaler.scale(loss_val).backward(retain_graph=True)
            loss_vals = routing_classification_loss_by_error(
                net, scaler, args.routing_error_quantile)
        for loss_val in loss_vals[:-1]:
            scaler.scale(loss_val).backward(retain_graph=True)
        scaler.scale(loss_vals[-1]).backward()
        if args.check_grads:
            for name, param in net.named_parameters():
                if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                    print('Bad grad in', name)
                    if args.hang_on_bad_grad:
                        breakpoint()
        if args.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(net.parameters(), args.clip_grad)  # pyright: ignore reportPrivateImportusage
        data_logger.record()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        batch_times.update(time.perf_counter() - end_time)
        end_time = time.perf_counter()

        if batch % args.print_freq == 0 and batch != 0:
            log.log(f'    [{batch}/{len(train_loader)}] '
                    f'Avg loss: {losses.mean():.5f} '
                    f'Avg time/batch: {batch_times.mean():.3f} s ')
    log.log(f'    **Train** Loss {losses.mean():.5f}')
    return losses.mean()


def validate(args, validation_loader, net, log, metrics, data_logger):
    """Validate on the given dataset."""
    net.eval()
    metrics.reset()
    with torch.no_grad():
        for samples, targets in validation_loader:
            samples = samples.to(get_cuda_device(), non_blocking=True)
            targets = targets.to(get_cuda_device(), non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.fp16):  # pyright: ignore reportPrivateImportusage
                output = net(samples)
                if data_logger is not None:
                    data_logger.record()
                    if 'preds' in args.save:
                        data_logger.record_tensor('preds', output)
                        data_logger.record_tensor('targets', targets)
                metrics.compute_metrics(output, targets)

            metrics.update_trackers(samples.size(0))
    metrics.log_metrics(log, indent=4, prefix='**Val**')
    return metrics.get_metric_means()


def main():
    """Manage training."""
    args = get_args()

    # Determine whether this is the primary rank.
    args.primary = get_world_rank(required=args.dist) == 0 or args.set_gpu_rank

    # Seed RNGs.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get job ID.
    if args.job_id is None:
        args.job_id = get_job_id()
        if args.job_id is None:
            raise RuntimeError('No job ID specified')

    if args.dist and args.set_gpu_rank:
        raise RuntimeError('Cannot use --dist and --set-gpu-rank')

    if args.dist:
        if get_local_size() > get_num_gpus():
            raise RuntimeError(
                'Do not use more ranks per node than there are GPUs')

        initialize_dist(f'./init_{args.job_id}', args.rendezvous)
    elif args.set_gpu_rank:
        torch.cuda.init()
        torch.cuda.set_device(get_local_rank(required=True))
    else:
        if get_world_size() > 1:
            print('Multiple processes detected, but --dist not passed',
                  flush=True)

    if not args.no_cudnn_bm and not args.deterministic:
        torch.backends.cudnn.benchmark = True

    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    # Set up output directory.
    if not args.output_dir:
        raise RuntimeError('Must specify --output-dir')
    if args.primary:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

    if args.early_stop and args.no_eval:
        raise RuntimeError('Must do validation to use early stopping')

    if args.primary:
        # Log arguments/configuration to a yaml file.
        with open(os.path.join(
                args.output_dir, f'args_{args.job_id}.yaml'), 'w') as f:
            yaml.dump(vars(args), f)

    log = Logger(os.path.join(args.output_dir, f'log_{args.job_id}.txt'),
                 args.primary, print_stdout=not args.no_print_stdout)

    # Set up data loaders.
    # Need to do this before the model, in case we need the data shape.
    train_loader, validation_loader, test_loader = get_dataloaders(args)
    num_train_samples = len(train_loader.dataset)  # pyright: ignore
    if args.no_eval or validation_loader is None:
        num_val_samples = 0
        num_test_samples = 0
    else:
        num_val_samples = len(validation_loader.dataset)  # pyright: ignore
        num_test_samples = len(test_loader.dataset)  # pyright: ignore

    # Gradient scaler for AMP.
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)  # pyright: ignore reportPrivateImportusage
    # So this can be accessed for unscaling.
    # Must come before model construction in case it needs to get a ref
    # to the scaler.
    args._grad_scaler = scaler

    # Set up the model.
    net = construct_model(args.model, args)
    net = net.to(get_cuda_device())
    net_without_ddp = net
    if args.dist:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True,
            bucket_cap_mb=args.bucket_cap)
        net_without_ddp = net.module
    # Initialize the model.
    smoe.models.init.init_params(args.initialization, net,
                                 exclude=['bn'],
                                 gate_orthogonal=args.gate_orthogonal)
    # Loss function.
    if args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'huber':
        criterion = torch.nn.HuberLoss(delta=args.huber_delta)
    elif args.loss == 'scaled-mse':
        if not args.mask:
            raise ValueError('scaled-mse loss requires a mask')
        if not args.loss_region_scales:
            raise ValueError('scaled-mse loss requires loss-region-scales')
        criterion = MaskScaledLoss(args.mask, args.loss_region_scales,
                                   F.mse_loss)
    elif args.loss == 'scaled-huber':
        if not args.mask:
            raise ValueError('scaled-mse loss requires a mask')
        if not args.loss_region_scales:
            raise ValueError('scaled-mse loss requires loss-region-scales')
        criterion = MaskScaledLoss(
            args.mask, args.loss_region_scales,
            functools.partial(F.huber_loss,
                              delta=args.huber_delta))
    elif args.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Unknown loss function {args.loss}')
    criterion = criterion.to(get_cuda_device())
    if args.rc_loss:
        # Currently only support with a single layer.
        if args.num_experts:
            raise RuntimeError('--rc-loss only works with single-layer SMoEs')
    # Optimizer.
    if args.optimizer == 'sgd':
        optimizer = apex.optimizers.FusedSGD(
            net.parameters(),
            args.lr,
            args.momentum)
    elif args.optimizer == 'adam':
        optimizer = apex.optimizers.FusedAdam(
            net.parameters(),
            args.lr,
            betas=(args.beta1, args.beta2))
    else:
        raise ValueError(f'Unknown optimizer {args.optimizer}')

    # Set up learning rate schedule.
    if args.schedule == 'plateau':
        if args.no_eval:
            raise ValueError('Cannot use plateau schedule without validation')
        if not args.plateau_epochs:
            raise ValueError('Must pass --plateau-epochs for plateau schedule')
        scheduler = ReduceLROnPlateauPatch(
            optimizer, factor=args.plateau_factor,
            patience=args.plateau_epochs,
            mode='max' if args.opt_metric_max else 'min')
    else:
        # Trivial scheduler that does not change the learning rate.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0)

    # Record the current best validation optimization metric (i.e., the
    # metric used for early stopping, etc.) and metrics during training.
    best_opt_metric = float('-inf') if args.opt_metric_max else float('inf')
    best_metrics = None
    # Number of epochs without an improvement in the validation loss.
    num_flat_epochs = 0

    # Estimate reasonable print frequency as every 5%.
    if args.print_freq is None:
        args.print_freq = max(len(train_loader) // 20, 1)

    # Log training configuration.
    log.log(str(args))
    log.log(str(net))
    log.log(f'Network has {count_parameters(net)} parameters')
    log.log(f'Using {get_world_size() if args.dist else 1} processes')
    log.log('Global batch size is'
            f' {args.batch_size*get_world_size() if args.dist else args.batch_size}'
            f' ({args.batch_size} per GPU)')
    log.log(f'Training dataset: {train_loader.dataset}')
    log.log(f'Training data size: {num_train_samples}')
    if args.no_eval or validation_loader is None:
        log.log('No validation')
    else:
        log.log(f'Validation dataset: {validation_loader.dataset}')
        log.log(f'Validation data size: {num_val_samples}')
        if test_loader is not None:
            log.log(f'Test dataset: {test_loader.dataset}')
            log.log(f'Test data size: {num_test_samples}')
    log.log(f'Starting learning rate: {args.lr}'
            f' | Target learning rate: {args.lr}')

    metrics = MetricManager(args.metric, criterion, num_val_samples,
                            prcntclose=args.prcntclose,
                            prcntclose_tol_scale=args.prcntclose_tol_scale,
                            allreduce=args.dist, mask=args.mask)
    data_logger = DataLogger(net, 'weights' in args.save, 'grads' in args.save)
    if 'routing' in args.save:
        data_logger.record_module_attribute('routing_map')
        data_logger.record_module_attribute('routing_weights')
    test_metrics = MetricManager(args.metric, criterion, num_test_samples,
                                 prcntclose=args.prcntclose,
                                 prcntclose_tol_scale=args.prcntclose_tol_scale,
                                 allreduce=args.dist, mask=args.mask)

    # Resume from checkpoint.
    if args.resume:
        log.log(f'Resuming from checkpoint {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        net_without_ddp.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_opt_metric = checkpoint['best_opt_metric']
        best_metrics = checkpoint['best_metrics']
        num_flat_epochs = checkpoint['num_flat_epochs']

    if args.eval_on_init and args.start_epoch == 0:
        log.log('Running validation on network initialization')
        validate(args, validation_loader, net, log, metrics, data_logger)
        data_logger.clear()

    # Train.
    log.log('Starting training at ' +
            time.strftime('%Y-%m-%d %X', time.gmtime(time.time())))
    epoch_times = AverageTracker()
    train_loss = float('inf')
    train_start_time = time.perf_counter()
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.perf_counter()
        if args.dist and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(epoch)  # pyright: ignore
            if not args.no_eval and validation_loader is not None:
                validation_loader.sampler.set_epoch(epoch)  # pyright: ignore
        log.log(f'==>> Epoch={epoch:03d}/{args.epochs:03d} '
                f'Elapsed={int(time.perf_counter()-train_start_time)} s '
                f'(avg epoch time: {epoch_times.mean():5.3f} s, current epoch: {epoch_times.latest():5.3f} s) '
                f'[learning_rate={scheduler.get_last_lr()[0]:6.4f}]')
        train_loss = train(args, train_loader, net, scaler, criterion,
                           optimizer, log, data_logger)
        new_best = False
        val_metric = None
        if not args.no_eval:
            metric_means = validate(args, validation_loader, net, log,
                                       metrics, data_logger)
            val_metric = metric_means[args.opt_metric]
            if args.opt_metric_max and val_metric > best_opt_metric:
                new_best = True
            elif not args.opt_metric_max and val_metric < best_opt_metric:
                new_best = True
            if new_best:
                best_opt_metric = val_metric
                best_metrics = metric_means
                num_flat_epochs = 0  # Reset early stopping counter.
                log.log(f'New best {args.opt_metric}: {best_opt_metric:.5f}')
            else:
                num_flat_epochs += 1
            if args.save_on_best and new_best:
                data_logger.save(args.job_id, args.output_dir, 'best')

        if not args.save_on_best:
            data_logger.save(args.job_id, args.output_dir, epoch)
        else:
            data_logger.clear()
        if args.schedule == 'plateau':
            scheduler.step(val_metric)
        else:
            scheduler.step()

        if hasattr(train_loader, 'reset'):
            train_loader.reset()  # pyright: ignore
        if not args.no_eval and validation_loader is not None:
            if hasattr(validation_loader, 'reset'):
                validation_loader.reset()  # pyright: ignore

        if args.primary:
            checkpoint = {
                'net': net_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_opt_metric': best_opt_metric,
                'best_metrics': best_metrics,
                'num_flat_epochs': num_flat_epochs
            }
            torch.save(checkpoint, os.path.join(
                args.output_dir, f'checkpoint_cur_{args.job_id}.pth'))
            if new_best:
                torch.save(checkpoint, os.path.join(
                    args.output_dir, f'checkpoint_best_{args.job_id}.pth'))

        epoch_times.update(time.perf_counter() - start_time)

        # Check if metric level reached and terminate early.
        if args.stop_on_metric_level is not None:
            if args.opt_metric_max:
                stop = val_metric >= args.stop_on_metric_level
            else:
                stop = val_metric <= args.stop_on_metric_level
            if stop:
                log.log(f'Stopping because {args.opt_metric} reached'
                        f' {args.stop_on_metric_level}')
                break

        # Terminate early if no improvement.
        if args.early_stop and num_flat_epochs >= args.early_stop:
            log.log('Stopping early after no improvement for '
                    f'{num_flat_epochs} epochs')
            break

    if not args.no_eval and test_loader is not None:
        test_metrics = validate(args, test_loader, net, log, test_metrics, None)
    else:
        test_metrics = None

    log.log(f'==>> Done Elapsed={int(time.perf_counter()-train_start_time)} s '
            f'(avg epoch time: {epoch_times.mean():5.3f} s '
            f'current epoch: {epoch_times.latest():5.3f} s)')
    log.log(f'     Final train loss: {train_loss:.5f}')
    metrics.log_metrics(log, indent=4, prefix='Best validation',
                        metrics=best_metrics)
    if test_metrics:
        metrics.log_metrics(log, indent=4, prefix='Test', metrics=test_metrics)


if __name__ == '__main__':
    main()
