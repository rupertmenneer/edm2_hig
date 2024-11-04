
import os
import zipfile
import json
import io
import PIL
import torch
import numpy as np
from torch_geometric.data import HeteroData, Dataset as GeoDataset
from training.dataset import Dataset
import h5py
from torch_geometric.data import Data, Batch
from hig_data.augmentations import HIGAugmentation
import copy

try:
    import pyspng
except ImportError:
    pyspng = None


"""
Models COCO with a HIG graph.
"""
# RUPERT MENNEER (2024)
#----------------------------------------------------------------------------
class CocoStuffGraphDataset(Dataset):

    def __init__(self,
        path,                           # Json file containing paths for this dataset
        latent_compression = 8,       # Compression factor for latent images.
        resolution = 512,             # Ensure specific resolution, None = anything goes.
        augmentation = False,         # whether to apply augmentation
        **kwargs,                     # Additional arguments for the GeoDataset base class.
    ) -> None:

        with open(path, 'r') as f:
            paths = json.load(f) 

        self._image_path = paths['image_path']
        self._mask_path = paths['mask_path']
        self._labels_path = paths['labels_path']
        self._captions_path = paths['captions_path']
        self._vocab_path = paths['vocab_path'] 
        
        self.file_names = self.get_filelist_from_paths()
        self.latent_compression = latent_compression

        # load shape
        name = os.path.splitext(os.path.basename(self._image_path))[0]
        raw_shape = [len(self._all_fnames['image']), 3, resolution, resolution]
        super().__init__(name=name, raw_shape=raw_shape, **kwargs) # Call Super class init

        # image node positions (same for every img)
        self.grid_size = resolution//latent_compression # grid size for image patches 
        self.num_image_nodes = self.grid_size * self.grid_size

        self.image_patch_positions = get_image_patch_positions(image_size=self.grid_size, patch_size=latent_compression)

        self.augmentation = None
        if augmentation:
            self.augmentation = HIGAugmentation(size=resolution)
            
    def get_filelist_from_dir(self, path):
        return {os.path.relpath(os.path.join(root, fname), start=path) for root, _dirs, files in os.walk(path) for fname in files}
        
    def get_filelist_from_paths(self,):
        
        assert os.path.isdir(self._image_path), 'Image path must point to a dir'
        assert os.path.isdir(self._mask_path), 'Mask path must point to a dir'
        assert self._file_ext(self._labels_path) == '.json', 'Label path must point to a json'

        supported_ext = {'.jpg', '.jpeg', '.png', '.npy'}
        unspported_prefix = ['.', '__', '__MACOSX/']
        self._files = {}  # Store file references
        primary_list = sorted(fname for fname in set(self.get_filelist_from_dir(self._image_path)) if self._file_ext(fname) in supported_ext and not any(fname.startswith(prefix) for prefix in unspported_prefix))
        file_sets = [primary_list,  self.get_filelist_from_dir(self._mask_path), self._get_jsonfile(self._labels_path).keys(), self.get_filelist_from_dir(self._captions_path)]
        complete_sets = self._extract_complete_suffix_set_files(file_sets)

        print('Found {} complete datapoint in {}'.format(len(complete_sets), self._image_path))
        
        self._all_fnames = {
            'image': sorted(f for f in set(self.get_filelist_from_dir(self._image_path)) if self._file_name(f) in complete_sets),
            'mask': sorted(f for f in set(self.get_filelist_from_dir(self._mask_path)) if self._file_name(f) in complete_sets),
            'label': sorted(f for f in set(self._get_jsonfile(self._labels_path).keys()) if self._file_name(f) in complete_sets),
            'caption': sorted(f for f in set(self.get_filelist_from_dir(self._captions_path)) if self._file_name(f) in complete_sets),
            'vocab': sorted(set(self._get_jsonfile(self._vocab_path).keys())),
        }
        assert len(self._all_fnames['image']) == len(self._all_fnames['mask']) == len(self._all_fnames['label']) == len(self._all_fnames['caption']), 'Number of files must match'
        

    # Function to extract unique IDs
    def _extract_complete_suffix_set_files(self, sets_of_file_lists):
        complete_sets = set()
        sets_to_check = [set(self._file_name(f) for f in l) for l in sets_of_file_lists]
        for i, file in enumerate(sets_of_file_lists[0]):  # list over first file list
            file_name = self._file_name(file)

            if file_name in complete_sets:
                continue
            # check this file exists in all other file lists
            if all(file_name in s for s in sets_to_check):  # Check all paths in one go
                complete_sets.add(file_name)
                
        return complete_sets


    def _file_name(self, fname):
        return os.path.splitext(os.path.basename(fname))[0]

    def _get_zipfile(self, path):
        if path not in self._files:
            self._files[path] = zipfile.ZipFile(path)
        return self._files[path]

    def _get_jsonfile(self, path):
        if path not in self._files:
            with open(path, 'r') as f:
                self._files[path] = json.load(f) 
        return self._files[path]
    

    def _open_file(self, fname, path):
        if self._file_ext(path) == '.zip':
            return self._get_zipfile(path).open(fname, 'r')
        return open(os.path.join(path, fname), 'rb')

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def _load_attribute_image(self, raw_idx, attribute='image', path=None):
        image_name = self._all_fnames[attribute][raw_idx]
        image = self._load_image_from_path(image_name, path)
        if attribute == 'mask' and image.ndim == 3:
            image = image[0:1]  # Use only the first channel
        return image
            
    # load the preprocessed image and mask from the coco dataset
    def _load_images(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._load_attribute_image(raw_idx, attribute='image', path=self._image_path)
        mask = self._load_attribute_image(raw_idx, attribute='mask', path=self._mask_path)
        assert isinstance(image, np.ndarray)
        assert isinstance(mask, np.ndarray)
        return image.copy(), mask.copy()
    
    def _get_class_latent(self, class_idx):
        label = self._get_jsonfile(self._vocab_path).get(str(class_idx+1))
        return label
    
    def _load_label(self, idx):
        raw_idx = self._raw_idx[idx]
        f_key = self._all_fnames['label'][raw_idx]
        label = self._get_jsonfile(self._labels_path)[f_key]
        return copy.copy(label)
    
    def _load_caption(self, idx):
        raw_idx = self._raw_idx[idx]
        fname = self._all_fnames['caption'][raw_idx]
        with self._open_file(fname, self._captions_path) as f:
            caption = torch.from_numpy(np.load(f))
        return caption

    # construct a hig from raw data item 
    def __getitem__(self, idx: int) -> HeteroData:

        data = RelaxedHeteroData() # create hetero data object for hig
        img, mask = self._load_images(idx)
        label = self._load_label(idx)
        if self.augmentation: # apply augmentation if available
            img, mask, label = self.augmentation(img, mask, label)
        caption = self._load_caption(idx)

        data.image = img[np.newaxis,...] # add image to data object
        data.mask = mask[np.newaxis,...] # add mask to data object
        data.caption = caption[np.newaxis,...] # add caption to data object
        data.name = self._all_fnames['mask'][self._raw_idx[idx]]
        data.labels = label

        # image and mask must have same resolution unless latent images enabled
        if not self.latent_compression != 1 and data.image.shape[-1] != data.mask.shape[-1]:
            raise IOError('Image and mask must have the same resolution if latent images are not enabled')

        # initialise image patch nodes
        image_patch_placeholder = torch.zeros((self.num_image_nodes, 1), dtype=torch.float32,)
        data['image_node'].x = image_patch_placeholder
        data['image_node'].pos = self.image_patch_positions

        # create class/instance nodes from semantic segmentation map/labels (bounding boxes)
        self._create_class_nodes(data, mask)
        self._create_instance_nodes(data, label, idx)
        self.connect_instance_and_class(data)

        return data
    
    def _create_instance_nodes(self, data, label, idx):
        
        class_labels = np.array([l-1 for l in label['obj_class'] if l != 183], dtype=np.int64) # exlcude other class
        class_latents = np.array([self._get_class_latent(l) for l in class_labels], dtype=np.float32)
        data['instance_node'].x = torch.from_numpy(class_latents)
        data['instance_node'].label = class_labels # add class labels for convenience

        edge_index = torch.combinations(torch.arange(len(class_labels)), with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data['instance_node', 'instance_edge', 'instance_node'].edge_index = edge_index

        data = self._create_instance_to_image_edges(data, label, idx)
        return data
    
    def _create_instance_to_image_edges(self, data, label, idx):
        edges = []
        bounding_boxes = [bb for i, bb in enumerate(label['obj_bbox']) if label['obj_class'][i] != 183] # exclude other class
        for class_node_idx, bounding_box in enumerate(bounding_boxes):
            # get bounding box coordinates
            linear_image_idxs = bbox_to_linear_indices(bounding_box, image_size=(self.grid_size, self.grid_size))
            node_id_repeated = np.full((len(linear_image_idxs),), class_node_idx, dtype=np.int64) # repeat class node idx for each patch
            edge_index = np.stack([node_id_repeated, linear_image_idxs], axis=0)
            edges.append(edge_index)

        class_to_image_index = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int64)
        data['instance_node', 'instance_to_image', 'image_node'].edge_index = torch.from_numpy(class_to_image_index) # convert to torch

        return data
    
    def _create_class_nodes(self, data, mask):
        class_labels = np.array([l for l in np.unique(mask) if l != 255], dtype=np.int64) # Must add 1 to harmonize mask labels with COCOStuff labels, remove unlabeled
        class_latents = np.array([self._get_class_latent(l) for l in class_labels], dtype=np.float32)
        data['class_node'].x = torch.from_numpy(class_latents)
        data['class_node'].label = class_labels # add class labels for convenience

        # densely connect class nodes
        edge_index = torch.combinations(torch.arange(len(class_labels)), with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data['class_node', 'class_edge', 'class_node'].edge_index = edge_index

        # create class to image edges from semantic segmentation map
        data = self._create_class_to_image_edges(data, mask)
        return data
    
    def _create_class_to_image_edges(self, data, mask):
        edges = []
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(self.grid_size, self.grid_size), mode='nearest').squeeze() # resize mask to match compression
        class_node_pos = []
        for class_node_idx, class_label in enumerate(data['class_node'].label):
            class_mask = np.argwhere(resized_mask == class_label) # get mask idxs for current class
            linear_image_patch_print_line_idxs = (class_mask[0] * self.grid_size + class_mask[1]).long() # linearise image node idxs
            avg_image_idxs = linear_to_avg_2d_idx(linear_image_patch_print_line_idxs, img_width=self.grid_size) if linear_image_patch_print_line_idxs.nelement() > 0 else np.zeros(2)
            class_node_pos.append(avg_image_idxs) # get image patch positions

            node_id_repeated = np.full((len(linear_image_patch_print_line_idxs),), class_node_idx, dtype=np.int64) # repeat class node idx for each patch
            edge_index = np.stack([node_id_repeated, linear_image_patch_print_line_idxs], axis=0)
            edges.append(edge_index)
        class_to_image_index = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int64)
        data['class_node', 'class_to_image', 'image_node'].edge_index = torch.from_numpy(class_to_image_index)
        data['class_node'].pos = np.stack(class_node_pos, axis=0) if class_node_pos else np.empty((0, 2), dtype=np.float32) # add class node positions for visualisation
        data['class_node'].pos = torch.from_numpy(data['class_node'].pos)
        return data
    
    def connect_instance_and_class(self, data):
        # Get the class labels and instance labels
        class_labels = data['class_node'].label
        instance_labels = data['instance_node'].label
        # Find where class labels match instance labels using broadcasting
        # This will create a matrix where each (i, j) is True if instance_label[j] == class_label[i]
        match_matrix = torch.tensor(class_labels[:, None] == instance_labels[None, :])
        # Get the indices where the labels match
        class_indices, instance_indices = match_matrix.nonzero(as_tuple=True)
        # Stack the indices into an edge index and assign it
        edge_index = torch.stack([instance_indices, class_indices], dim=0)
        data['instance_node', 'instance_to_class', 'class_node'].edge_index = edge_index
        return data

# ----------------------------------------------------------------------------
import re

class RelaxedHeteroData(HeteroData):
    # modified to allow allow_empty flag to be passed
    def __getattr__(self, key: str, allow_empty=True):
        if hasattr(self._global_store, key):
            return getattr(self._global_store, key)
        elif bool(re.search('_dict$', key)):
            return self.collect(key[:-5], allow_empty=allow_empty)
        raise AttributeError(f"'{self.__class__.__name__}' has no "
                             f"attribute '{key}'")
    

def get_image_patch_positions(image_size = 32, patch_size = 8) -> torch.Tensor:
    """
    Get the xy positions of each image patch in the image grid
    """
    grid_size = image_size
    grid_h = torch.arange(grid_size, dtype=torch.float32,)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    image_patch_positions = torch.stack(grid, axis=0).flatten(1).permute(1, 0)  # (num_patches, 2) 
    image_patch_positions = image_patch_positions * patch_size # scale to full res size without compression
    return image_patch_positions

def linear_to_avg_2d_idx(idx, img_width=32, patch_size=8):
    row = torch.mean((idx // img_width).float()) * patch_size
    col = torch.mean((idx % img_width).float()) * patch_size
    return torch.tensor([col, row])

def linear_to_2d_idx(idx, img_width=32, patch_size=1):
    row = (idx // img_width).float() * patch_size
    col = (idx % img_width).float() * patch_size
    return torch.stack([row, col]).long()

def bbox_to_linear_indices(bbox, image_size):
    """
    Converts normalized bounding box coordinates to linear pixel indices within the image.

    Parameters:
    - bbox: list or tuple of normalized coordinates [xmin, ymin, xmax, ymax], values between 0 and 1.
    - image_size: tuple (H, W), image height and width.

    Returns:
    - linear_indices: numpy array of linear indices of pixels within the bounding box.
    """
    xmin, ymin, xmax, ymax = bbox
    H, W = image_size

    # Convert normalized coordinates to pixel indices
    xmin_pix = int(np.floor(xmin * W))
    xmax_pix = int(np.ceil(xmax * W)) - 1
    ymin_pix = int(np.floor(ymin * H))
    ymax_pix = int(np.ceil(ymax * H)) - 1

    # Ensure indices are within image bounds
    xmin_pix = max(0, min(xmin_pix, W - 1))
    xmax_pix = max(0, min(xmax_pix, W - 1))
    ymin_pix = max(0, min(ymin_pix, H - 1))
    ymax_pix = max(0, min(ymax_pix, H - 1))

    # Generate grid of x and y indices
    ys = np.arange(ymin_pix, ymax_pix + 1)
    xs = np.arange(xmin_pix, xmax_pix + 1)
    xs_grid, ys_grid = np.meshgrid(xs, ys)

    # Flatten the grids and compute linear indices
    x_indices = xs_grid.flatten()
    y_indices = ys_grid.flatten()
    linear_indices = y_indices * W + x_indices

    return linear_indices
