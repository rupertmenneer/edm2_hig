# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion models according to the EDM2 recipe from the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop_hignn_control
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'edm2-img512-xs':   dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-s':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-m':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-l':    dnnlib.EasyDict(duration=1792<<20, batch=2048, channels=320, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xl':   dnnlib.EasyDict(duration=1280<<20, batch=2048, channels=384, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxl':  dnnlib.EasyDict(duration=896<<20,  batch=2048, channels=448, lr=0.0065, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img64-s':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=192, lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-m':     dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-l':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=320, lr=0.0080, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-xl':    dnnlib.EasyDict(duration=640<<20,  batch=2048, channels=384, lr=0.0070, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-coco256-s':   dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-coco256-xs2':  dnnlib.EasyDict(duration=2048<<20, batch=1024, channels=128, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-coco256-s2':  dnnlib.EasyDict(duration=2048<<20, batch=512, channels=192, lr=0.0070, decay=70000, dropout=0.20, P_mean=-0.4, P_std=1.0),
    'edm2-coco256-s3':  dnnlib.EasyDict(duration=2048<<20, batch=512, channels=192, lr=0.0100, decay=70000, dropout=0.10, P_mean=-0.3, P_std=1.0),
    'edm2-coco256-xs3':  dnnlib.EasyDict(duration=2048<<20, batch=128, channels=128, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.3, P_std=1.0),
    'edm2-coco256-m':    dnnlib.EasyDict(duration=2048<<20, batch=512, channels=256, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-vg512-m':    dnnlib.EasyDict(duration=2048<<20, batch=512, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-vg512-s':    dnnlib.EasyDict(duration=2048<<20, batch=512, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-vg512-l':    dnnlib.EasyDict(duration=896<<20, batch=512, channels=320, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='edm2-img512-s',
                          dataset_name = 'hig_data.vg2.VGGraphPrecomputedDataset',
                          **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()
    
    # Preset.
    
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Dataset.
    c.dataset_kwargs = dnnlib.EasyDict(class_name=dataset_name, path=opts.path)
    c.val_dataset_kwargs = dnnlib.EasyDict(class_name=dataset_name, path=opts.val_path, split='val')
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.val_dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')


    # Hyperparameters.
    c.update(total_nimg=opts.duration, batch_size=opts.batch)
    c.network_kwargs = dnnlib.EasyDict(class_name='training.networks_edm2_hignn_control_xs.Precond', model_channels=opts.channels, dropout=opts.dropout)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop_hignn_control.EDM2Loss', P_mean=opts.P_mean, P_std=opts.P_std)
    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop_hignn_control.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay)
    c.wandb_kwargs = dnnlib.EasyDict(project='COCO_edm2_hig', mode='online', id=opts.wandb_id)
    
    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = opts.get('fp16', True)
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # I/O-related options.
    c.status_nimg = opts.get('status', 0) or None
    c.snapshot_nimg = opts.get('snapshot', 0) or None
    c.checkpoint_nimg = opts.get('checkpoint', 0) or None
    c.wandb_nimg = opts.get('wandb_vis', 0) or None
    c.seed = opts.get('seed', 0)
    c.preset_name = preset
    c.cond = opts.get('cond', True)

    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {False}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop_hignn_control.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=str, default='edm2-img512-s', show_default=True)

# COCO
@click.option('--path',             help='Path to the dataset', metavar='H5|ZIP|DIR',              type=str, required=True)
@click.option('--val_path',         help='Path to the dataset', metavar='H5|ZIP|DIR',              type=str, required=True)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)

# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NIMG',           type=parse_nimg, default='256Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NIMG',       type=parse_nimg, default='8Mi', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',    type=parse_nimg, default='16Mi', show_default=True)
@click.option('--wandb_vis',        help='Interval of wandb vis', metavar='NIMG',               type=parse_nimg, default='4Mi', show_default=True)
@click.option('--wandb_id',         help='idx of wandb run to resume', metavar='NIMG',          type=str, required=False, default=None)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=7, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

def cmdline(outdir, dry_run, **opts):
    """Train diffusion models according to the EDM2 recipe from the paper
    "Analyzing and Improving the Training Dynamics of Diffusion Models".

    Examples:

    \b
    # Train XS-sized model for ImageNet-512 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train_edm2.py \\
        --outdir=training-runs/00000-edm2-img512-xs \\
        --data=datasets/img512-sd.zip \\
        --preset=edm2-img512-xs \\
        --batch-gpu=32

    \b
    # To resume training, run the same command again.
    """
    torch.multiprocessing.set_start_method('spawn',force=True)
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=outdir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
