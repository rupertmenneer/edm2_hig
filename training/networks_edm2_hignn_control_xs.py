# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# ------- MODIFIED FROM -------
# Improved diffusion model architecture proposed in the paper Analyzing
# and Improving the Training Dynamics of Diffusion Models.

"Graph-Conditioned Diffusion Models for Image Synthesis"

import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc
from training.networks_hignn_attn import HIGnnInterface
import copy

#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a, b, t=0.5):
    if isinstance(t, torch.Tensor):
        return a.lerp(b, t) / torch.sqrt((1 - t) ** 2 + t ** 2)
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

@persistence.persistent_class
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))
    

#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

@persistence.persistent_class
class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb, c_emb=None):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        # if c_emb is not None:
            # c_emb = torch.nn.functional.interpolate(c_emb, size=(y.shape[-2:]), mode='nearest')
            # y = mp_sum(y, c_emb.to(y.dtype), t=self.res_balance)
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x
    

class ControlZeroConv(MPConv):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, kernel=[1,1])
        self.gain = torch.nn.Parameter(torch.zeros([]))
        # torch.nn.init.zeros_(self.weight)

    def forward(self, x):
        return super().forward(x, gain=self.gain)
    
    

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = [3,3,3,3],    # MODIFICATION: List of residual blocks per resolution instead of default 3 per res.
        attn_resolutions    = [16, 8],      # List of resolutions with self-attention.
        gnn_metadata        = None,         # MODIFICATION: Metadata for dual gnn
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        control_net_type    = 'mp_sum',
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        self.model_channels = model_channels
        self.img_channels = img_channels
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        print(f"Control XS strat with control modulation")
        self.gnn_metadata = gnn_metadata

        self.control_net_type = control_net_type

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.control_emb_label = MPConv(label_dim, 1000, kernel=[]) if label_dim != 0 else None # use pre-trained label network
        torch.nn.init.zeros_(self.control_emb_label.weight)
        self.emb_label = MPConv(1000, cemb, kernel=[]) if label_dim != 0 else None
        

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
                self.init_conv = f'{res}x{res}_conv'
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks[level]):
                cin = cout
                cout = channels
                self.enc[f'control_cross_{res}x{res}_block{idx}'] = ControlZeroConv(cin, cin) # add a zero conv operator before every res block
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        bottleneck_cin = cin
        bottleneck_cout = cout
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for name, block in self.enc.items() if 'control' not in name]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks[level] + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3])

        # ControlNet.
        self.hignn = HIGnnInterface(gnn_metadata, gnn_channels=model_channels, num_gnn_layers=3, **block_kwargs)
        self.control_e = copy.deepcopy(self.enc)
        self.control_d = torch.nn.ModuleDict()
        skips = [block.out_channels for name, block in self.enc.items() if 'control' not in name]
        cin = bottleneck_cin
        cout = bottleneck_cout
        for level, _ in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.control_d[f'bottleneck'] = ControlZeroConv(cin, cin)
            for idx in range(num_blocks[level] + 1):
                cin = skips.pop()
                self.control_d[f'{res}x{res}_block{idx}'] = ControlZeroConv(cin, cin)

    def apply_conditional_control(self, control_x, graph, control_modulation):
        if graph is not None:
            hig_out, graph = self.hignn(control_x, graph=graph) # MODIFICATION: encode input with hignn and graph

            if self.control_net_type == 'mp_sum':
                control_x = mp_sum(control_x, hig_out.to(control_x.dtype), t=control_modulation)

            elif self.control_net_type == 'mult':
                hig_out = hig_out + 1
                control_x = mp_silu(control_x * hig_out.to(control_x.dtype))

            elif self.control_net_type == 'cross_attn':
                control_x = self.cross_attention(control_x, hig_out.to(control_x.dtype))    

        return control_x, hig_out, graph

    def forward(self, x, noise_labels, graph, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            class_labels = mp_silu(self.control_emb_label(class_labels)) # apply new control layer to case to dim-1000 before applying existing
            emb = mp_sum(emb, self.emb_label(class_labels), t=self.label_balance)
        emb = mp_silu(emb)
        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

        skips, control_skips = [], []
        control_x = x
        c_emb = None
        for (name, block), (c_name, c_block) in zip(self.enc.items(), self.control_e.items()):
            if 'control_cross' in name: # if zero conv control block, cross modulate between enc and control enc
                control_x = mp_sum(control_x, c_block(x), t=self.label_balance) # add current enc block features to control net encoder
                x = mp_sum(x, block(control_x), t=self.label_balance) # add current control net block features to regular encoder
            else:
                x = block(x) if 'conv' in name else block(x, emb)
                control_x = c_block(control_x) if 'conv' in name else c_block(control_x, emb,)
                if self.init_conv in name: # for init conv apply conditional control
                    control_x, c_emb, graph = self.apply_conditional_control(control_x, graph, self.label_balance)
                skips.append(x)
                control_skips.append(control_x)

        # Decoder.
        control_bottleneck = self.control_d[f'bottleneck'](control_x)
        x = mp_sum(x, control_bottleneck, t=self.label_balance)
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, mp_sum(skips.pop(), self.control_d[name](control_skips.pop()), t=self.label_balance), t=self.concat_balance)

            x = block(x, emb)

        x = self.out_conv(x, gain=self.out_gain)
        return x
    
    
    def cross_attention(self, x, y):
        # Cross-attention.
        assert self.num_heads != 0

        # Compute Q from x.
        q = self.attn_q(x)
        q = q.reshape(q.shape[0], self.num_heads, -1, q.shape[2] * q.shape[3])
        q = normalize(q, dim=2)  # pixel norm
        
        # Compute K and V from y.
        k, v = self.attn_kv(y).split(2, dim=1)
        k = k.reshape(k.shape[0], self.num_heads, -1, k.shape[2] * k.shape[3])
        v = v.reshape(v.shape[0], self.num_heads, -1, v.shape[2] * v.shape[3])
        k, v = normalize(k, dim=2), normalize(v, dim=2)  # pixel norm

        # Compute attention weights.
        w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
        # Compute attention output.
        y = torch.einsum('nhqk,nhck->nhcq', w, v)
        
        # Project back to original shape.
        y = self.attn_proj(y.reshape(*x.shape))
        
        # Combine x and y (cross-attention result).
        x = mp_sum(x, y, t=self.attn_balance)
        return x


#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

@persistence.persistent_class
class Precond(torch.nn.Module):
    def __init__(self,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        label_dim       = 0,    # Class label dimensionality. 0 = unconditional.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = UNet(img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim, **unet_kwargs)
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, graph=None, class_labels=None, force_fp32=False, return_logvar=False, **unet_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        graph = graph.clone().to(x.device) if graph is not None else graph # cast graph to device (and clone to avoid in place errors during sampling)

        # turn caption to class labels input
        if class_labels is None:
            caption = None if graph is None or not hasattr(graph, 'caption') else graph.caption.to(x.device)
            class_labels = None if self.label_dim == 0 else torch.zeros([x.shape[0], self.label_dim], device=x.device) if caption is None else caption.to(torch.float32).reshape(-1, self.label_dim).clone().to(x.device)
        assert not torch.isnan(class_labels).any(), "NaN detected in 'caption'"

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
        x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, noise_labels=c_noise, graph=graph, class_labels=class_labels, **unet_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x
    
    def init_control_weights(self, load_pretrained=True):
        if load_pretrained:
            self.unet.control_e.load_state_dict(self.unet.enc.state_dict(), strict=False) # load pretrained encoder weights to control enc
        for name, param in self.named_parameters():
            if any(keyword in name for keyword in ['control', 'hignn', 'emb_label']):
                param.requires_grad = True
            else:
                param.requires_grad = False

#----------------------------------------------------------------------------


