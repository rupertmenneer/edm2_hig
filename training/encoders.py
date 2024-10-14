# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Converting between pixel and latent representations of image data."""

import os
import warnings
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc

warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.
#
# Logically, "raw pixels" are first encoded into "raw latents" that are
# then further encoded into "final latents". Decoding, on the other hand,
# goes directly from the final latents to raw pixels. The final latents are
# used as inputs and outputs of the model, whereas the raw latents are
# stored in the dataset. This separation provides added flexibility in terms
# of performing just-in-time adjustments, such as data whitening, without
# having to construct a new dataset.
#
# All image data is represented as PyTorch tensors in NCHW order.
# Raw pixels are represented as 3-channel uint8.

@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into [-1, +1].

@persistence.persistent_class
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x): # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)

#----------------------------------------------------------------------------
# Pre-trained VAE encoder from Stability AI.

# ---- ImageNet (SD)
#  raw_mean    = [5.81, 3.25, 0.12, -2.15]
#  raw_std     = [4.17, 4.62, 3.71, 3.28]

# ---- COCO-stuff train2017 (SDXL)
# raw_mean  = [6.56,  1.10,  0.10, -0.31]
# raw_std   = [[4.63, 4.33, 5.51, 4.26]

# ----  Visual Genome

@persistence.persistent_class
class StabilityVAEEncoder(Encoder):
    def __init__(self,
        vae_name    = 'stabilityai/sdxl-vae',       # Name of the VAE to use.
        raw_mean    = [6.56,  1.10,  0.10, -0.31],  # Assumed mean of the raw latents.
        raw_std     = [4.63, 4.33, 5.51, 4.26], # Assumed standasrd deviation of the raw latents.
        final_mean  = 0,                            # Desired mean of the final latents.
        final_std   = 0.5,                          # Desired standard deviation of the final latents.
        batch_size  = 8,                            # Batch size to use when running the VAE.
    ):
        super().__init__()
        self.vae_name = vae_name
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._vae = None

    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_stability_vae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def _run_vae_encoder(self, x):
        d = self._vae.encode(x)['latent_dist']
        return torch.cat([d.mean, d.std], dim=1)

    def _run_vae_decoder(self, x):
        return self._vae.decode(x)['sample']

    def encode_pixels(self, x): # raw pixels => raw latents
        self.init(x.device)
        x = x.to(torch.float32) / 255
        x = torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])
        return x

    def encode_latents(self, x): # raw latents => final latents
        mean, std = x.to(torch.float32).chunk(2, dim=1)
        x = mean + torch.randn_like(mean) * std
        x = x * misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        return x

    def decode(self, x): # final latents => raw pixels
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = x.clamp(0, 1).mul(255).to(torch.uint8)
        return x

#----------------------------------------------------------------------------

def load_stability_vae(vae_name='stabilityai/sdxl-vae', device=torch.device('cpu')):
    import dnnlib
    cache_dir = dnnlib.make_cache_dir_path('diffusers')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HOME'] = cache_dir

    import diffusers # pip install diffusers # pyright: ignore [reportMissingImports]
    try:
        # First try with local_files_only to avoid consulting tfhub metadata if the model is already in cache.
        vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name, cache_dir=cache_dir, local_files_only=True)
    except:
        # Could not load the model from cache; try without local_files_only.
        vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name, cache_dir=cache_dir)
    return vae.eval().requires_grad_(False).to(device)

#----------------------------------------------------------------------------

# CLIP encoder to retreive text for text captions/class labels/attributes etc.

@persistence.persistent_class
class CLIPEncoder(Encoder):
    def __init__(self,
        name        = 'openai/clip-vit-large-patch14',        # Name of the VAE to use.
        raw_mean    = [6.56,  1.10,  0.10, -0.31],            # Assumed mean of the raw latents.
        raw_std     = [5.315, 4.646, 5.719, 4.495],           # Assumed standasrd deviation of the raw latents.
        final_mean  = 0,                                      # Desired mean of the final latents.
        final_std   = 0.5,                                    # Desired standard deviation of the final latents.
        batch_size  = 8,                                      # Batch size to use when running the VAE.
    ):
        super().__init__()
        self.name = name
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._clip = None
        self._processor = None

    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._clip is None:
            self._clip, self._processor = load_clip(self.name, device=device)
        else:
            self._clip.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _clip=None, _processor=None) # do not pickle the vae

    def _run_clip_encoder(self, inputs):
        latents = self._clip.get_text_features(**inputs)
        return latents # return avg text embedding per input

    def encode_raw_text(self, text, device=None): # list of raw text => raw latents
        self.init(device)
        inputs = self._processor(text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()} # cast inputs to device
        return self._run_clip_encoder(inputs)

    def encode_latents(self, x): # raw latents => final latents
        x = x.to(torch.float32)
        x = x * misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        return x
    
#----------------------------------------------------------------------------

def load_clip(name='openai/clip-vit-large-patch14', device=torch.device('cpu')):
    import dnnlib
    cache_dir = dnnlib.make_cache_dir_path('transformers')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HOME'] = cache_dir

    import transformers # pip install diffusers # pyright: ignore [reportMissingImports]
    try:
        # First try with local_files_only to avoid consulting tfhub metadata if the model is already in cache.
        clip = transformers.CLIPModel.from_pretrained(name, cache_dir=cache_dir, local_files_only=True)
        processor = transformers.CLIPProcessor.from_pretrained(name, cache_dir=cache_dir, local_files_only=True)
    except:
        # Could not load the model from cache; try without local_files_only.
        clip = transformers.CLIPModel.from_pretrained(name, cache_dir=cache_dir,)
        processor = transformers.CLIPProcessor.from_pretrained(name, cache_dir=cache_dir,)

    return clip.eval().requires_grad_(False).to(device), processor