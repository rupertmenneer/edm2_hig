import torchvision
import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset as GeoDataset

from graph_datasets.hf_dataset import HuggingFaceDataset
from training.dataset import Dataset
import os
import zipfile
import PIL

try:
    import pyspng
except ImportError:
    pyspng = None

"""
Loads COCO dataset from HF.
"""
# RUPERT MENNEER (2024)
#----------------------------------------------------------------------------

class COCOStuffDataset(Dataset):
    
    def __init__(self,
        image_path,                   # Path to directory or zip for images.
        mask_path,                    # Path to directory or zip for semantic masks.
        resolution = None,            # Ensure specific resolution, None = anything goes.
        **super_kwargs,               # Additional arguments for the Dataset base class.
        ):

        self._path = image_path
        self._mask_path = mask_path
        self._zipfiles = {}  # Store zipfile references for images and masks
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._cached_masks = dict() # {raw_idx: np.ndarray, ...}

        if self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = {
                'image': set(self._get_zipfile(image_path).namelist()),
                'mask': set(self._get_zipfile(mask_path).namelist())
            }
            assert len(self._all_fnames['image']) == len(self._all_fnames['mask']), 'Different number of images and masks'
        else:
            raise IOError('Path must point to a zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        unspported_prefix = ['.', '__', '__MACOSX/']
        self._fnames = {
            'image': sorted(fname for fname in self._all_fnames['image'] if self._file_ext(fname) in supported_ext and not any(fname.startswith(prefix) for prefix in unspported_prefix)),
            'mask': sorted(fname for fname in self._all_fnames['mask'] if self._file_ext(fname) in supported_ext and not any(fname.startswith(prefix) for prefix in unspported_prefix)),
        }
        if len(self._fnames['image']) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._fnames['image'])] + list(self._load_attribute_image(0, path=self._path).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self, path):
        if path not in self._zipfiles:
            self._zipfiles[path] = zipfile.ZipFile(path)
        return self._zipfiles[path]

    def _open_file(self, fname, path):
        if self._type == 'zip':
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

    def _load_image_from_path(self, fname, path): # return image from fname as np.ndarray
        ext = self._file_ext(fname)
        with self._open_file(fname, path) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_attribute_image(self, raw_idx, attribute='image', path=None):
        image_name = self._fnames[attribute][raw_idx]
        return self._load_image_from_path(image_name, path)

        # load the preprocessed image and mask from the coco dataset
    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        mask = self._cached_masks.get(raw_idx, None)
        if image is None:
            image = self._load_attribute_image(raw_idx, attribute='image', path=self._path)
            mask = self._load_attribute_image(raw_idx, attribute='mask', path=self._mask_path)
            if self._cache:
                self._cached_images[raw_idx] = image
                self._cached_masks[raw_idx] = mask
        assert isinstance(image, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
            mask = mask[:, :, ::-1]
        return image.copy(), mask.copy()


#----------------------------------------------------------------------------

"""
Loads COCO dataset from HF.
"""
# RUPERT MENNEER (2024)
#----------------------------------------------------------------------------

class HFCOCOStuffDataset(HuggingFaceDataset):

    def __init__(self, dataset_name='limingcv/Captioned_COCOStuff', split='train', **super_kwargs,):
        super().__init__(dataset_name, split=split, **super_kwargs)

        self.img_tfm = self.get_standard_image_transform()
        self.mask_tfm = self.get_standard_image_transform(interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._cached_masks = dict() # {raw_idx: np.ndarray, ...}
        self.image_label_column = 'image'
        self.mask_label_column = 'panoptic_seg_map'

    # process the image and mask, mask is resized with nearest neighbour interpolation
    def _preprocess(self, data):
        data[self.image_label_column] = np.array(self.img_tfm(data[self.image_label_column])).transpose(2, 0, 1) # rgb img
        data[self.mask_label_column] = self.mask_tfm(torch.tensor(data[self.mask_label_column], dtype=torch.int16).unsqueeze(0)).numpy() # mask
        return data
    
    # load the preprocessed image and mask from the coco dataset
    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        mask = self._cached_masks.get(raw_idx, None)
        if image is None:
            data = self._preprocess(self.dataset[int(raw_idx)])
            image = data[self.image_label_column]
            mask = data[self.mask_label_column]
            if self._cache:
                self._cached_images[raw_idx] = image
                self._cached_masks[raw_idx] = mask
        assert isinstance(image, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
            mask = mask[:, :, ::-1]
        return image.copy(), mask.copy()
    
    def _load_raw_labels(self):
        return self.dataset.labels

"""
Models data with a dual graph representation.
"""
# RUPERT MENNEER (2024)
#----------------------------------------------------------------------------
class CocoStuffGraphDataset(GeoDataset):

    def __init__(self,
        image_path,                   # Path to directory or zip for images.
        mask_path,                    # Path to directory or zip for semantic masks.
        graph_transform=None,         # Transform to apply to the graph.
        patch_size = 8,               # Size of image patches. Set to 1 for no patches.
        n_labels = 182,               # Number of classes in the dataset.
    ) -> None:

        super().__init__(None, graph_transform)
        self.dataset = COCOStuffDataset(image_path, mask_path)
        self.n_labels = n_labels

        # image node positions (same for every img)
        self.grid_size = self.dataset._raw_shape[-1] // patch_size
        self.num_image_nodes = self.grid_size * self.grid_size

        self.image_patch_positions = get_image_patch_positions(self.dataset._raw_shape[-1], patch_size)
        

    def __len__(self) -> int:
        return len(self.dataset)

    # construct a dual graph from raw data item 
    def __getitem__(self, idx: int) -> HeteroData:

        data = RelaxedHeteroData() # create hetero data object for dual graph
        img, mask = self.dataset[idx]
        data.image = torch.from_numpy(img[np.newaxis,...]) # add image to data object
        data.mask = torch.from_numpy(mask[np.newaxis,...]) # add mask to data object

        # initialise image patch nodes
        image_patch_placeholder = torch.zeros(self.num_image_nodes, 1, dtype=torch.float32)
        data['image_node'].x = image_patch_placeholder
        data['image_node'].pos = self.image_patch_positions

        # create class nodes from semantic segmentation map
        self._create_class_nodes(data, mask)

        # create class to image edges from semantic segmentation map
        self._create_class_to_image_edges(data, mask)

        if self.transform: # apply any additional transforms
            data = self.transform(data)

        return data
    
    def _create_class_nodes(self, data, mask):
        class_labels = np.array([l for l in np.unique(mask) if l != 255], dtype=np.int16)
        print(class_labels)
        onehots = np.zeros((len(class_labels), self.n_labels), dtype=np.float32)
        onehots[np.arange(len(class_labels)), class_labels] = 1
        onehots = np.concatenate([class_labels[...,np.newaxis], onehots], axis=1) # add class labels to onehots position 0 for convenience
        data['class_node'].x = torch.from_numpy(onehots).to(torch.float32)
        # densely connect class nodes
        
        edge_index = torch.combinations(torch.arange(len(class_labels)), with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data['class_node', 'class_edge', 'class_node'].edge_index = edge_index
        return data
    
    def _create_class_to_image_edges(self, data, mask):
        edges = []
        resized_mask = torch.nn.functional.interpolate(torch.tensor(mask[np.newaxis,...], dtype=torch.float32), size=(self.grid_size, self.grid_size), mode='nearest').squeeze() # resize mask to match compression
        class_node_pos = []
        for class_node_idx, class_label in enumerate(data['class_node'].x[:,0]):
            class_mask = np.argwhere(resized_mask == class_label) # get mask idxs for current class
            linear_image_patch_print_line_idxs = (class_mask[0] * self.grid_size + class_mask[1]).long() # linearise image node idxs

            class_node_pos.append(linear_to_avg_2d_idx(linear_image_patch_print_line_idxs)) # get image patch positions

            node_id_repeated = np.full((len(linear_image_patch_print_line_idxs),), class_node_idx, dtype=np.int16) # repeat class node idx for each patch
            edge_index = np.stack([node_id_repeated, linear_image_patch_print_line_idxs], axis=0)
            edges.append(edge_index)
        class_to_image_index = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int16)
        data['class_node', 'class_to_image', 'image_node'].edge_index = torch.from_numpy(class_to_image_index) # convert to torch
        data['class_node'].pos = torch.stack(class_node_pos, dim=0) # add class node positions for visualisation
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
    

def get_image_patch_positions(image_size = 256, patch_size = 8) -> torch.Tensor:
    """
    Get the xy positions of each image patch in the image grid
    """
    grid_size = image_size // patch_size
    grid_h = torch.arange(grid_size, dtype=torch.float32,)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    image_patch_positions = torch.stack(grid, axis=0).flatten(1).permute(1, 0)  # (num_patches, 2) 
    image_patch_positions *= patch_size
    return image_patch_positions

def linear_to_avg_2d_idx(idx, img_width=32, patch_size=8):
    row = torch.mean((idx // img_width).float())*patch_size
    col = torch.mean((idx % img_width).float())*patch_size
    return torch.tensor([col, row])