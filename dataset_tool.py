# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

from collections.abc import Iterator
from dataclasses import dataclass
import functools
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import h5py

from training.encoders import StabilityVAEEncoder, CLIPEncoder
from hig_data.coco2 import CocoStuffGraphDataset, CocoStuffGraphDatasetLightweight
from hig_data.utils import DataLoader


#----------------------------------------------------------------------------

@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]
    fname: Optional[str]

#----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    input_images = []
    def _recurse_dirs(root: str): # workaround Path().rglob() slowness
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))
    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    # Load labels.
    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    # No labels available => determine from top-level directory names.
    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            try:
                img = np.array(PIL.Image.open(fname).convert('RGB'))
                yield ImageEntry(img=img, label=labels.get(arch_fnames[fname]), fname=fname)
            except PIL.UnidentifiedImageError: # catch image file errors
                    print(f"Skipping corrupted image: {fname}")
                    continue
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        # Load labels.
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                try:
                    with z.open(fname, 'r') as file:
                        img = np.array(PIL.Image.open(file).convert('RGB'))
                    yield ImageEntry(img=img, label=labels.get(fname), fname=fname)
                except PIL.UnidentifiedImageError: # catch image file errors
                    print(f"Skipping corrupted image: {fname}")
                    continue
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

def open_json(source):
    data = dict()
    with open(source) as f:
        data  = json.load(f)
    return data

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img, 'RGB')
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    def center_crop_imagenet(image_size: int, arr: np.ndarray):
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
        """
        pil_image = PIL.Image.fromarray(arr)
        while min(*pil_image.size) >= 2 * image_size:
            new_size = tuple(x // 2 for x in pil_image.size)
            assert len(new_size) == 2
            pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

        scale = image_size / min(*pil_image.size)
        new_size = tuple(round(x * scale) for x in pil_image.size)
        assert len(new_size) == 2
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    def nearest_crop(image_size: int, arr: np.ndarray):
        """
        Center cropping for semantic masks using NEAREST resampling to preserve label information.
        """
        pil_image = PIL.Image.fromarray(arr)

        # Calculate scale based on the target image size
        scale = image_size / min(*pil_image.size)
        new_size = tuple(round(x * scale) for x in pil_image.size)
        assert len(new_size) == 2

        # Resize the image using NEAREST resampling
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.NEAREST)

        # Convert back to numpy array and crop the center
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    if transform == 'center-crop-dhariwal':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        if output_width != output_height:
            raise click.ClickException('width and height must match in --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_imagenet, output_width)
    if transform == 'nearest-crop': 
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        if output_width != output_height:
            raise click.ClickException('width and height must match in --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(nearest_crop, output_width)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        if file_ext(source) == 'json':
            return open_json(source,)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal', 'nearest-crop']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)

# MODIFIED - Now supports instance bounding box jsons, and captions
def convert(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]]
):
    """Convert an image dataset into archive format for training.

    Specifying the input images:

    \b
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, class labels are determined from
    top-level directory names.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    The --transform=center-crop-dhariwal selects a crop/rescale mode that is intended
    to exactly match with results obtained for ImageNet in common diffusion model literature:

    \b
    python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \\
        --dest=datasets/img64.zip --resolution=64x64 --transform=center-crop-dhariwal
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    transform_image = make_transform(transform, *resolution if resolution is not None else (None, None))
    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):

        archive_fname = f'{os.path.splitext(os.path.basename(image.fname))[0]}.png'

        # Apply crop and resize.
        img = transform_image(image.img)
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        assert img.ndim == 3
        cur_image_attrs = {'width': img.shape[1], 'height': img.shape[0]}
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                raise click.ClickException(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if width != 2 ** int(np.floor(np.log2(width))):
                raise click.ClickException('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            raise click.ClickException(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img)
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image.label] if image.label is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--model-url',  help='VAE encoder model', metavar='URL',                  type=str, default='stabilityai/sdxl-vae', show_default=True)
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--xflip',       help='Compute ONLY flipped', type=bool, required=True)

def encode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    xflip: bool,
):
    """Encode pixel data to VAE latents."""
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=1)
    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    labels = []

    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        img_tensor = torch.tensor(image.img).to('cuda').permute(2, 0, 1).unsqueeze(0)
        if xflip:
            img_tensor = img_tensor.flip(-1) # horiz flip -> we must do this offline as flipped latents don't produce flipped images
        mean_std = vae.encode_pixels(img_tensor)[0].cpu()
        archive_fname = f'{os.path.splitext(os.path.basename(image.fname))[0]}.npy'

        f = io.BytesIO()
        np.save(f, mean_std)
        save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())
        labels.append([archive_fname, image.label] if image.label is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--model-url',  help='Text encoder model', metavar='URL', type=str, default='openai/clip-vit-large-patch14', show_default=True)
@click.option('--source',     help='Input json file', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--key',       help='Output directory or archive name', metavar='PATH',  type=str, required=False, default='caption',)

def textencode(
    model_url: str,
    source: str,
    dest: str,
    key: str = 'caption',

):
    """Encode pixel data to VAE latents."""
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    clip = CLIPEncoder(name=model_url, batch_size=1)
    text_data = open_dataset(source, max_images=None)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    for idx, data in tqdm(text_data.items(), total=len(text_data)):

        text_data = data[key]
        text_latents = clip.encode_raw_text(text_data, device='cuda')[0].cpu()

        archive_fname = f'{os.path.splitext(os.path.basename(idx))[0]}.npy'
        f = io.BytesIO()
        np.save(f, text_latents)
        save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())

    close_dest()


#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--img_path',     help='Input imgs latents', metavar='PATH',   type=str, required=True)
@click.option('--image_path_flipped',  help='Input latents flipped', metavar='PATH',   type=str, required=True)
@click.option('--mask_path',     help='Input masks', metavar='PATH',   type=str, required=True)
@click.option('--label_path',     help='Input bounding boxes', metavar='PATH',   type=str, required=True)
@click.option('--caption_path',     help='Input caption latents', metavar='PATH',   type=str, required=True)
@click.option('--vocab_path',     help='Input vocab latents', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--xflip',       help='Compute flipped', type=bool, required=True)

def graphencodecoco(
    img_path: str,
    image_path_flipped: str,
    mask_path: str,
    label_path: str,
    caption_path: str,
    vocab_path: str,
    dest: str,
    xflip: bool,
):
    """Encode graph dataset to compressed precomputed form."""
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')
    assert os.path.splitext(dest)[1].lower() == '.h5', 'graph encode expects output dir to be h5 format'

    dataset = CocoStuffGraphDataset(img_path, mask_path, labels_path=label_path, captions_path=caption_path, vocab_path=vocab_path, image_path_flipped=image_path_flipped, xflip=xflip)
    statistics = {k:0 for k in ['img_mean', 'img_std', 'cap_mean', 'cap_std']}

    with h5py.File(dest, 'w') as hdf:
        for idx in tqdm(range(len(dataset)), total=len(dataset)):

            # get graph and file name
            graph = dataset[idx]
            raw_idx = dataset._raw_idx[idx]
            image_filename = dataset._file_name(dataset._all_fnames['image'][raw_idx])
            out_fname = image_filename
            if xflip:
                is_flipped = dataset._xflip[idx]
                out_fname = f"{image_filename}-f" if is_flipped else image_filename
                
            # create group for datapoint
            group = hdf.create_group(out_fname)

            # Store image/mask/caption arrays
            group.create_dataset('image', data=graph.image, compression="gzip")
            group.create_dataset('mask', data=graph.mask, compression="gzip")
            group.create_dataset('caption', data=graph.caption, compression="gzip") 

            # Store graph components as separate datasets to a sub group
            graph_group = group.create_group('graph')
            graph_group.create_dataset('instance_node', data=graph['instance_node'].x, compression="gzip")
            graph_group.create_dataset('instance_label', data=graph['instance_node'].label, compression="gzip")

            graph_group.create_dataset('class_node', data=graph['class_node'].x, compression="gzip")
            graph_group.create_dataset('class_pos', data=graph['class_node'].pos, compression="gzip")
            graph_group.create_dataset('class_label', data=graph['class_node'].label, compression="gzip")

            graph_group.create_dataset('class_edge', data=graph['class_edge'].edge_index, compression="gzip")
            graph_group.create_dataset('instance_edge', data=graph['instance_edge'].edge_index, compression="gzip")
            
            graph_group.create_dataset('class_to_image', data=graph['class_to_image'].edge_index, compression="gzip")
            graph_group.create_dataset('instance_to_image', data=graph['instance_to_image'].edge_index, compression="gzip")

            # Statistics
            statistics['img_mean'] += graph.image.mean(axis=(-2,-1))
            statistics['img_std'] += graph.image.std(axis=(-2,-1))
            statistics['cap_mean'] += graph.caption.mean()
            statistics['cap_std'] += graph.caption.std()

        # Save statistics
        statistics = {k:v/len(dataset) for k,v in statistics.items()}
        group = hdf.create_group('statistics')
        group.create_dataset('img_mean', data=statistics['img_mean'], compression="gzip")
        group.create_dataset('img_std', data=statistics['img_std'], compression="gzip")
        group.create_dataset('cap_mean', data=np.array(statistics['cap_mean']),) 
        group.create_dataset('cap_std', data=np.array(statistics['cap_std']),) 


#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--path',     help='Input imgs latents', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--repeats',       help='How many repeats', type=int, required=True)
@click.option('--augmentation',       help='Apply aug?', type=bool, required=True)
@click.option('--batch_size',       help='Apply aug?', type=int, required=False)

def graphencodecoco2(
    path: str,
    dest: str,
    repeats: int = 2,
    augmentation: bool = True,
    batch_size: int = 32,
):
    """Encode graph dataset to compressed precomputed form."""
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')
    assert os.path.splitext(dest)[1].lower() == '.h5', 'graph encode expects output dir to be h5 format'

    dataset = CocoStuffGraphDatasetLightweight(path)
    statistics = {k:0 for k in ['img_mean', 'img_std']}
    vae = StabilityVAEEncoder(batch_size=batch_size)
    dataloader = DataLoader(dataset, augmentation=augmentation, batch_size=batch_size, pin_memory=True, num_workers=6, prefetch_factor=6)
    
    with h5py.File(dest, 'w') as hdf:

        for repeat_idx in range(repeats):
            for (img, mask, label) in tqdm(dataloader, total=len(dataloader)):
                
                # convert img->latent
                img_tensor = img.to('cuda')
                mean_std = vae.encode_pixels(img_tensor).cpu()

                # store mask at latent size
                resized_mask = torch.nn.functional.interpolate(mask, size=(dataset.grid_size, dataset.grid_size), mode='nearest').squeeze() # resize mask to match compression

                # Store image/mask/caption/bounding box arrays to h5
                for i in range(img.shape[0]):
                    image_filename = f'{label[i]['filename']}_{repeat_idx}'
                    group = hdf.create_group(image_filename) # create group for datapoint
                    group.create_dataset('image', data=mean_std[i], compression="gzip")
                    group.create_dataset('mask', data=resized_mask[i], compression="gzip")
                    group.create_dataset('obj_bbox', data=label[i]['obj_bbox'], compression="gzip") 
                    group.create_dataset('obj_class', data=label[i]['obj_class'], compression="gzip") 
                
                # Statistics

                statistics['img_mean'] += np.array(mean_std).mean(axis=(0,-2,-1))
                statistics['img_std'] += np.array(mean_std).std(axis=(0,-2,-1))

        # Save statistics
        statistics = {k:v/len(dataloader)*repeats for k,v in statistics.items()}
        group = hdf.create_group('statistics')
        group.create_dataset('img_mean', data=statistics['img_mean'], compression="gzip")
        group.create_dataset('img_std', data=statistics['img_std'], compression="gzip")


#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--model-url',  help='VAE encoder model', metavar='URL',                  type=str, default='stabilityai/sdxl-vae', show_default=True)
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)

def decode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
):
    """Decode VAE latents to pixels."""
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=1)
    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    labels = []

    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        std_mean = image.img
        assert isinstance(std_mean, np.ndarray)
        lat = torch.tensor(std_mean).unsqueeze(0).cuda()
        pix = vae.decode(vae.encode_latents(lat))[0].permute(1, 2, 0).cpu().numpy()
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        img = PIL.Image.fromarray(pix, 'RGB')
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image.label] if image.label is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
