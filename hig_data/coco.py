
import os
import zipfile
import json

import PIL
import torch
import numpy as np
from torch_geometric.data import HeteroData, Dataset as GeoDataset
from training.dataset import Dataset

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
        image_path,                   # Path to zip for images.
        mask_path,                    # Path to zip for semantic masks.
        labels_path,                  # Path to json for bounding boxes.
        captions_path,                # Path to json for caption latents.
        vocab_path,                   # Path to json for vocab latents.
        latent_compression = 8,       # Compression factor for latent images.
        resolution = 32,              # Ensure specific resolution, None = anything goes.
        **kwargs,                     # Additional arguments for the GeoDataset base class.
    ) -> None:

        self._image_path = image_path
        self._mask_path = mask_path
        self._labels_path = labels_path
        self._captions_path = captions_path
        self._vocab_path = vocab_path 
        
        self.file_names = self.get_filelist_from_paths()
        self.latent_compression = latent_compression

        # load shape
        name = os.path.splitext(os.path.basename(self._image_path))[0]
        raw_shape = [len(self._all_fnames['image'])] + list(self._load_attribute_image(0, path=self._image_path).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **kwargs) # Call Super class init

        # image node positions (same for every img)
        self.grid_size = self._raw_shape[-1] # grid size for image patches 
        self.num_image_nodes = self.grid_size * self.grid_size
        self.image_patch_positions = get_image_patch_positions()
            
        
    def get_filelist_from_paths(self,):
        
        assert self._file_ext(self._image_path) == '.zip', 'Image path must point to a zip'
        assert self._file_ext(self._mask_path) == '.zip', 'Mask path must point to a zip'
        assert self._file_ext(self._labels_path) == '.json', 'Label path must point to a json'

        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        unspported_prefix = ['.', '__', '__MACOSX/']
        self._files = {}  # Store file references

        primary_list = sorted(fname for fname in set(self._get_zipfile(self._image_path).namelist()) if self._file_ext(fname) in supported_ext and not any(fname.startswith(prefix) for prefix in unspported_prefix))
        file_sets = [primary_list, self._get_zipfile(self._mask_path).namelist(), self._get_jsonfile(self._labels_path).keys(), self._get_zipfile(self._captions_path).namelist()]
        complete_sets = self._extract_complete_suffix_set_files(file_sets)

        print('Found {} complete datapoint in {}'.format(len(complete_sets), self._image_path))
        
        self._all_fnames = {
            'image': sorted(f for f in set(self._get_zipfile(self._image_path).namelist()) if self._file_name(f) in complete_sets),
            'mask': sorted(f for f in set(self._get_zipfile(self._mask_path).namelist()) if self._file_name(f) in complete_sets),
            'label': sorted(f for f in set(self._get_jsonfile(self._labels_path).keys()) if self._file_name(f) in complete_sets),
            'caption': sorted(f for f in set(self._get_zipfile(self._captions_path).namelist()) if self._file_name(f) in complete_sets),
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
        return None

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
        image = self._load_attribute_image(idx, attribute='image', path=self._image_path)
        mask = self._load_attribute_image(idx, attribute='mask', path=self._mask_path)
        assert isinstance(image, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
            mask = mask[:, :, ::-1]
        return image.copy(), mask.copy()
    
    def _get_class_latent(self, class_idx):
        label = self._get_jsonfile(self._vocab_path).get(str(class_idx+1))
        return label
    
    def _load_label(self, idx):
        f_key = self._all_fnames['label'][idx]
        label = self._get_jsonfile(self._labels_path)[f_key]
        return label
    
    def _load_caption(self, idx):
        fname = self._all_fnames['caption'][idx]
        with self._open_file(fname, self._captions_path) as f:
            caption = np.load(f)
        return caption

    # construct a hig from raw data item 
    def __getitem__(self, idx: int) -> HeteroData:

        data = RelaxedHeteroData() # create hetero data object for hig
        raw_idx = self._raw_idx[idx]
        img, mask = self._load_images(raw_idx)
        label = self._load_label(raw_idx)
        caption = self._load_caption(raw_idx)

        data.image = img[np.newaxis,...] # add image to data object
        data.mask = mask[np.newaxis,...] # add mask to data object
        data.caption = caption[np.newaxis,...] # add caption to data object

        # image and mask must have same resolution unless latent images enabled
        if not self.latent_compression != 1 and data.image.shape[-1] != data.mask.shape[-1]:
            raise IOError('Image and mask must have the same resolution if latent images are not enabled')

        # initialise image patch nodes
        image_patch_placeholder = np.zeros((self.num_image_nodes, 1), dtype=np.float32)
        data['image_node'].x = image_patch_placeholder
        data['image_node'].pos = self.image_patch_positions

        # create class/instance nodes from semantic segmentation map/labels (bounding boxes)
        self._create_class_nodes(data, mask)
        self._create_instance_nodes(data, label)

        return data
    
    def _create_instance_nodes(self, data, label):
        
        class_labels = np.array([l-1 for l in label['obj_class'] if l != 183], dtype=np.int64) # exlcude other class
        class_latents = np.array([self._get_class_latent(l) for l in class_labels], dtype=np.float32)
        data['instance_node'].x = class_latents
        data['instance_node'].label = class_labels # add class labels for convenience

        edge_index = torch.combinations(torch.arange(len(class_labels)), with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data['instance_node', 'instance_edge', 'instance_node'].edge_index = edge_index.numpy()

        data = self._create_instance_to_image_edges(data, label)
        return data
    
    def _create_instance_to_image_edges(self, data, label):
        edges = []
        bounding_boxes = [bb for i, bb in enumerate(label['obj_bbox']) if label['obj_class'][i] != 183] # exclude other class
        for class_node_idx, bounding_box in enumerate(bounding_boxes):
            # get bounding box coordinates
            linear_image_idxs = bbox_to_linear_indices(bounding_box, image_size=(self.grid_size, self.grid_size))
            node_id_repeated = np.full((len(linear_image_idxs),), class_node_idx, dtype=np.int64) # repeat class node idx for each patch
            edge_index = np.stack([node_id_repeated, linear_image_idxs], axis=0)
            edges.append(edge_index)
        class_to_image_index = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int64)
        data['instance_node', 'instance_to_image', 'image_node'].edge_index = class_to_image_index # convert to torch

        return data
    
    def _create_class_nodes(self, data, mask):
        class_labels = np.array([l for l in np.unique(mask) if l != 255], dtype=np.int64) # Must add 1 to harmonize mask labels with COCOStuff labels, remove unlabeled
        class_latents = np.array([self._get_class_latent(l) for l in class_labels], dtype=np.float32)
        data['class_node'].x = class_latents
        data['class_node'].label = class_labels # add class labels for convenience

        # densely connect class nodes
        edge_index = torch.combinations(torch.arange(len(class_labels)), with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data['class_node', 'class_edge', 'class_node'].edge_index = edge_index.numpy()

        # create class to image edges from semantic segmentation map
        data = self._create_class_to_image_edges(data, mask)
        return data
    
    def _create_class_to_image_edges(self, data, mask):
        edges = []
        resized_mask = torch.nn.functional.interpolate(torch.tensor(mask[np.newaxis,...], dtype=torch.float32), size=(self.grid_size, self.grid_size), mode='nearest').squeeze() # resize mask to match compression
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
        data['class_node', 'class_to_image', 'image_node'].edge_index = class_to_image_index 
        data['class_node'].pos = np.stack(class_node_pos, axis=0) if class_node_pos else np.empty((0, 2), dtype=np.float32) # add class node positions for visualisation
        return data
    

# ----------------------------------------------------------------------------

# variant of COCOStuffGraph to load precomputes np files including graphs
class COCOStuffGraphPrecomputedDataset(GeoDataset):
    def __init__(self,
                path,                       # Path to directory or zip for images/masks/graphs.
                graph_transform = None,     # Transform to apply to the graph.
                max_size    = None,         # Maximum number of items to load.
                random_seed = 0,            # Random seed to use when applying max_size.
                cache       = True,         # Cache images in CPU memory?
                **kwargs,                   
        ) -> None:

        super().__init__(None, graph_transform)
        
        self._path = path
        self._cache = cache
        assert os.path.isdir(self._path), "Path must be a directory"

        if cache: # cache images, masks and graphs
            self._cached_images = dict() # {raw_idx: np.ndarray, ...}
            self._cached_masks = dict() # {raw_idx: np.ndarray, ...}
            self._cached_graphs = dict() # {raw_idx: np.ndarray, ...}
            self._cached_idxs = dict()

        self._unfiltered_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        self._suffixes = ['image.npy', 'mask.npy', 'graph.npz'] # coco graph suffixes
        self._data_fnames = sorted(self._extract_complete_suffix_set_files(self._unfiltered_fnames, self._suffixes))
        print('Found {} complete datapoint in {}'.format(len(self._data_fnames), self._path))
        self._name = os.path.splitext(os.path.basename(self._path))[0]

        # Apply max_size.
        dataset_size = len(self._data_fnames)
        self._raw_idx = np.arange(dataset_size, dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        raw_ref_image,*_ = self._load_coco_files(0)
        self._raw_shape = [dataset_size] + list(raw_ref_image.shape)

        self.grid_size = self._raw_shape[-1] # grid size for image patches 
        self.num_image_nodes = self.grid_size * self.grid_size
        self.image_patch_positions = get_image_patch_positions()
        

    # Function to extract unique IDs
    def _extract_complete_suffix_set_files(self, file_paths, suffixes):
        complete_sets = set()
        for i, file in enumerate(file_paths):  # list over all files
            if file.endswith(suffixes[0]): # check if file has primary suffix
                file_prefix = file[:-len(suffixes[0])]
                if file_prefix in complete_sets:
                    continue
                # Generate all potential file paths once
                potential_files = [file_prefix + suffix for suffix in suffixes]
                if all(f in self._unfiltered_fnames for f in potential_files):  # Check all paths in one go
                    complete_sets.add(os.path.basename(file_prefix))
        return complete_sets

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
        
    def _load_np_from_path(self, fname):  # return image from fname as np.ndarray or .npz content
        path = os.path.join(self._path, fname)
        ext = self._file_ext(path)
        if ext == '.npz':
            with np.load(path, allow_pickle=True) as arr:
                return dict(arr)  # Return the .npz file as a dictionary
        else:
            return np.load(path, allow_pickle=True)  # Return .npy file directly

    def __len__(self) -> int:
        return len(self._raw_idx)
    
    def _load_coco_files(self, idx):
        raw_idx = self._raw_idx[idx]

        if self._cache and raw_idx in self._cached_idxs:
            return self._cached_images[raw_idx], self._cached_masks[raw_idx], self._cached_graphs[raw_idx]

        img = self._load_np_from_path(self._data_fnames[raw_idx] + self._suffixes[0])
        # mask = self._load_np_from_path(self._data_fnames[raw_idx] + self._suffixes[1])
        mask_path = os.path.join(self._path, self._data_fnames[raw_idx]) + self._suffixes[1]
        graph = self._load_np_from_path(self._data_fnames[raw_idx] + self._suffixes[2])

        if self._cache:
            self._cached_images[raw_idx] = img
            self._cached_masks[raw_idx] = mask_path # store path to mask as only used for vis
            self._cached_graphs[raw_idx] = graph
            self._cached_idxs[raw_idx] = True

        return img, mask_path, graph

    # construct a hig from raw data item 
    def __getitem__(self, idx: int) -> HeteroData:

        data = RelaxedHeteroData() # create hetero data object for hig

        img, mask_path, graphs = self._load_coco_files(idx)

        data.image = torch.from_numpy(img[np.newaxis,...]).to(torch.float32) # add image to data object
        # data.mask = torch.from_numpy(mask[np.newaxis,...]).to(torch.int64) # add mask to data object
        data.mask_path = mask_path # add mask to data object

        # ---- IMAGE
        image_patch_placeholder = torch.empty((data.image.shape[-2]*data.image.shape[-1], 1), dtype=torch.float32) # must init be same size for correct minibatching
        data['image_node'].x = image_patch_placeholder
        data['image_node'].pos = self.image_patch_positions

        # ---- Class
        one_hots = torch.from_numpy(graphs['class_node']).to(torch.float32)
        data['class_node'].x = self.clean_coco_one_hots(one_hots) # move 'unlabelled' to the end
        data['class_node'].pos = self.safe_key_pos_open(graphs, 'class_pos')
        data['class_node'].label = torch.argmax(data['class_node'].x, dim=1)

        # ---- Edges
        data['class_node', 'class_edge', 'class_node'].edge_index = torch.from_numpy(graphs['class_edge']).to(torch.long) if graphs['class_edge'].shape[1] != 0 else torch.empty((2, 0), dtype=torch.long)
        data['class_node', 'class_to_image', 'image_node'].edge_index = torch.from_numpy(graphs['class_to_image']).to(torch.long) if graphs['class_to_image'].shape[1] != 0 else torch.empty((2, 0), dtype=torch.long)
        # add reverse image edge
        data['image_node', 'image_to_class', 'class_node'].edge_index = torch.flip(data['class_node', 'class_to_image', 'image_node'].edge_index, [0])

        return data
    
    def clean_coco_one_hots(self, one_hots, real_n_labels=183):
        cleaned_one_hots = torch.zeros((one_hots.shape[0], real_n_labels), dtype=torch.float32)
        cleaned_one_hots[:, :real_n_labels-2] = one_hots[:, :real_n_labels-2]
        cleaned_one_hots[:, real_n_labels-1] = one_hots[:, 255] # move 'unlabelled' to the end
        return cleaned_one_hots

    def safe_key_pos_open(self, graphs, key):
        class_pos = graphs.get(key, None)
        if class_pos is None or class_pos.size == 0 or class_pos.shape != (len(class_pos), 2):
            return torch.empty((0, 2), dtype=torch.float32)
        else:
            return torch.from_numpy(class_pos).to(torch.float32)
        
    @property
    def name(self):
        return self._name

    @property
    def image_shape(self): # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]
    
    def __getstate__(self):
        return dict(self.__dict__,)


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