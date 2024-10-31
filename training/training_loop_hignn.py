# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import torch.distributed
import torch.distributed
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
import wandb
from datetime import datetime
os.environ["WANDB_DISABLE_GPU"] = "true"
os.environ['WANDB_CACHE_DIR'] = '/home/rfsm2/rds/hpc-work/edm2_hig/wandb'

from hig_data.visualisation import logging_generate_sample_vis
from generate_images import edm_sampler

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5, cond_mean=-0.4, cond_std=1.0,):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.cond_mean = cond_mean
        self.cond_std = cond_std

    def __call__(self, net, images, graph=None, labels=None):
        
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma

        # MODIFICATION: cond noise
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        cond_sigma = (rnd_normal * self.cond_mean + self.cond_std).exp()

        denoised, logvar = net(images + noise, sigma=sigma, cond_sigma=cond_sigma, graph=graph, class_labels=labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=5):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr


#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='hig_data.coco.COCOStuffGraphPrecomputedDataset',),
    val_dataset_kwargs  = dict(class_name='hig_data.coco.COCOStuffGraphPrecomputedDataset',),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='hig_data.utils.DataLoader', pin_memory=True, num_workers=4, prefetch_factor=4),
    network_kwargs      = dict(class_name='training.networks_edm2_hignn.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop_hignn.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop_hignn.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),
    wandb_kwargs        = dict(project='COCO_edm2_hig', mode='online',), 

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 1024,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<14,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.
    wandb_nimg          = 128<<14,  # Wandb Vis every N training images. None = disable.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    preset_name         = None,    # Name of the preset for logging.

    device              = torch.device('cuda'),
    cfg_dropout         = 0.2,      # dropout chance of having 0 conditions (CFG)
    node_subsample      = True,     # uniform dropout of cond nodes
    cond                = True,     # conditional embeddings e.g. caption embs
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False    
    
    # Get current date and time
    now = datetime.now()
    formatted_date_time = now.strftime("%b_%d_hr_%H")
    ckpt_name = f"{preset_name}_{formatted_date_time}"
    if wandb_kwargs['mode'] != 'disabled' and dist.get_rank() == 0: # init wandb if not already
        wandb.init(**wandb_kwargs, name=f"{preset_name}_bs_{batch_size}_seed_{seed}", resume="allow")
        dist.print0(f"wandb init with ID {wandb_kwargs.get('id', None)}")

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)
    dist.print0(f"True adjusted batch per GPU {batch_gpu} for {num_accumulation_rounds} accumulation rounds.")

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    val_dataset_obj = dnnlib.util.construct_class_by_name(**val_dataset_kwargs)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_graph = dataset_obj[0]
    ref_image = ref_graph.image
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device))
    dist.print0('Constructing network...')

    # Network
    network_kwargs['label_dim'] = 768 if cond else 0 # add emd dim if conditional
    dist.print0(f"Label dim set to {network_kwargs['label_dim']}")
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], gnn_metadata=ref_graph.metadata())
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().to(device)
    inputs = [
            torch.zeros([1, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([1,], device=device),
            ref_graph.to(device),
            torch.zeros([1, net.label_dim], device=device),
        ]
    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, inputs, max_nesting=2)
    outputs = net(*inputs) # must pass through inputs for lazy initisation (on all ranks)
    net.train().requires_grad_(True).to(device)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], find_unused_parameters=True)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    # randomly dropout conditioning nodes uniformly
    dist.print0(f'Node subsampling: {node_subsample}')
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, subsample=node_subsample, **data_loader_kwargs))

    val_dataset_sampler = torch.utils.data.DistributedSampler(dataset=val_dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, shuffle=False,) # use standard sampler for val
    val_dataloader = dnnlib.util.construct_class_by_name(dataset=val_dataset_obj, batch_size=batch_gpu, sampler=val_dataset_sampler, **data_loader_kwargs, drop_last=True) # set drop last true
    
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None


    # create logging/validation batches/dls
    dist.print0(f"Single batch logging")
    logging_batch = get_single_batch(dataset_obj, class_name=data_loader_kwargs['class_name'], subsample=node_subsample)
    logging_batch_val = get_single_batch(val_dataset_obj, class_name=data_loader_kwargs['class_name'])

    dist.print0(f"Training loop starting...")
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Validation 
        misc.set_random_seed(seed)
        if wandb_nimg is not None and (done or state.cur_nimg % wandb_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            # Calculate val loss
            losses = misc.AverageMeter('Loss')
            val_iter = iter(val_dataloader)
            ddp.eval()
            with torch.no_grad():
                graph_batch = next(val_iter).to(device, non_blocking=True)
                image_latents = encoder.encode_latents(graph_batch.image.to(device))
                dist.print0(f"Validation batch -> ", image_latents.shape)
                loss = loss_fn(net=ddp, images = image_latents, graph=graph_batch)
                losses.update(torch.mean(loss).detach().item())
            ddp.train()
            losses.all_reduce()
            if dist.get_rank() == 0 and wandb_kwargs['mode'] != 'disabled': # save val loss to wandb only
                wandb.log({"val/loss": losses.avg, "nimg": state.cur_nimg})
            del val_iter # save memory
            # Sample and save to wandb
            dist.print0(f"Starting sampling..")
            if wandb_kwargs['mode'] != 'disabled': # save val loss to wandb only 
                with torch.no_grad():
                    for name, batch in zip(['Train', 'Validation'], [logging_batch, logging_batch_val]):            
                        graph = copy.deepcopy(batch) # ensure deepcopy of logging batch each call
                        b,_,h,w = graph.image.shape
                        sample_shape = (b, net.unet.img_channels, h, w)
                        noise = torch.randn(sample_shape, device=device)
                        dist.print0(f"{name} Sampling with image shape {noise.shape}")
                        sampled = edm_sampler(net=net, noise=noise, graph=graph) # sample images from noise and graph batch

                        # Create HIGNN embedding for logging
                        zero_input = torch.zeros((b, net.unet.model_channels,h,w), device=device)
                        hignn_out, _ = net.unet.enc['32x32_block0'].hignn(zero_input, graph=graph.to(device))
                        hignn_out = np.clip(hignn_out[:, :3].cpu().detach().numpy().transpose(0,2,3,1), 0, 1) # clip for vis
                        dist.print0(f"Logging {name} samples to wandb..")
                        if dist.get_rank() == 0: # save vis to rank 0 only
                            logging_generate_sample_vis(graph, sampled, hignn_out, title=name) # log images to wandb
            dist.print0(f"Validation Finished.")

        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = ckpt_name+f'-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, ckpt_name+f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()

        apply_cfg = False # if applying CFG must ensure consistency across ranks for graph gradients (e.g. they should all have conds or none)
        if cfg_dropout != 0.0:
            misc.set_random_seed(seed, state.cur_nimg)
            apply_cfg = np.random.rand() < cfg_dropout

        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                graph_batch = next(dataset_iterator).to(device)
                image_latents = encoder.encode_latents(graph_batch.image.to(device))
                graph_batch = None if apply_cfg else graph_batch
                labels = None if graph_batch is None else graph_batch.caption
                loss = loss_fn(net=ddp, images = image_latents, graph=graph_batch, labels=labels)
                training_stats.report('Loss/loss', loss)
                if dist.get_rank() == 0 and wandb.run is not None:
                        wandb.log({"train/loss": torch.mean(loss).detach(), "nimg": state.cur_nimg})
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        if dist.get_rank() == 0 and wandb.run is not None:
            wandb.log({"learning_rate": lr, "nimg": state.cur_nimg})
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time


#----------------------------------------------------------------------------

@torch.no_grad()
def get_single_batch(dataset, class_name, n=8, subsample=False):
    
    # create data loader, get a single batch, detach tensors, and return
    data_loader = dnnlib.util.construct_class_by_name(
        class_name=class_name,
        dataset=dataset,
        batch_size=n,
        pin_memory=True,
        subsample=subsample
    )
    
    # Retrieve a single batch, detaching the tensors after loading
    batch = next(iter(data_loader))
        
    return batch


 