
import os
import zipfile
import PIL
import torchvision
import torch
import numpy as np
from torch_geometric.data import HeteroData, Dataset as GeoDataset

from training.dataset import Dataset

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

        if len(self._fnames['image']) != len(self._fnames['mask']):
            raise IOError('Different number of images and masks')

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
        image = self._load_image_from_path(image_name, path)
        if attribute == 'mask' and image.ndim == 3:
            image = image[0:1]  # Use only the first channel
        return image
            

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
Models data with a dual graph representation.
"""
# RUPERT MENNEER (2024)
#----------------------------------------------------------------------------
class CocoStuffGraphDataset(GeoDataset):

    def __init__(self,
        image_path,                   # Path to directory or zip for images.
        mask_path,                    # Path to directory or zip for semantic masks.
        graph_transform=None,         # Transform to apply to the graph.
        n_labels = 182,               # Number of classes in the dataset.
        latent_compression = 8,       # Compression factor for latent images.
        **kwargs,                     # Additional arguments for the GeoDataset base class.
    ) -> None:

        super().__init__(None, graph_transform)
        
        self.dataset = COCOStuffDataset(image_path, mask_path)
        self.n_labels = n_labels
        self.latent_compression = latent_compression

        # image node positions (same for every img)
        self.grid_size = self.dataset._raw_shape[-1] # grid size for image patches 
        self.num_image_nodes = self.grid_size * self.grid_size

        self.image_patch_positions = get_image_patch_positions()
            
        

    def __len__(self) -> int:
        return len(self.dataset)

    # construct a hig from raw data item 
    def __getitem__(self, idx: int) -> HeteroData:

        data = RelaxedHeteroData() # create hetero data object for hig
        img, mask = self.dataset[idx]

        data.image = torch.from_numpy(img[np.newaxis,...]) # add image to data object
        data.mask = torch.from_numpy(mask[np.newaxis,...]) # add mask to data object

        # image and mask must have same resolution unless latent images enabled
        if not self.latent_compression != 1 and data.image.shape[-1] != data.mask.shape[-1]:
            raise IOError('Image and mask must have the same resolution if latent images are not enabled')

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
        class_labels = np.array([l for l in np.unique(mask) if l != 255], dtype=np.int64)
        onehots = np.zeros((len(class_labels), self.n_labels), dtype=np.float32)
        onehots[np.arange(len(class_labels)), class_labels] = 1

        data['class_node'].x = torch.from_numpy(onehots).to(torch.float32)
        data['class_node'].label = class_labels # add class labels to onehots position 0 for convenience

        # densely connect class nodes
        edge_index = torch.combinations(torch.arange(len(class_labels)), with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data['class_node', 'class_edge', 'class_node'].edge_index = edge_index
        return data
    
    def _create_class_to_image_edges(self, data, mask):
        edges = []
        resized_mask = torch.nn.functional.interpolate(torch.tensor(mask[np.newaxis,...], dtype=torch.float32), size=(self.grid_size, self.grid_size), mode='nearest').squeeze() # resize mask to match compression
        class_node_pos = []
        for class_node_idx, class_label in enumerate(data['class_node'].label):
            class_mask = np.argwhere(resized_mask == class_label) # get mask idxs for current class
            linear_image_patch_print_line_idxs = (class_mask[0] * self.grid_size + class_mask[1]).long() # linearise image node idxs
            avg_image_idxs = linear_to_avg_2d_idx(linear_image_patch_print_line_idxs, img_width=self.grid_size) if linear_image_patch_print_line_idxs.nelement() > 0 else torch.zeros(2)
            class_node_pos.append(avg_image_idxs) # get image patch positions

            node_id_repeated = np.full((len(linear_image_patch_print_line_idxs),), class_node_idx, dtype=np.int64) # repeat class node idx for each patch
            edge_index = np.stack([node_id_repeated, linear_image_patch_print_line_idxs], axis=0)
            edges.append(edge_index)
        class_to_image_index = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int64)
        data['class_node', 'class_to_image', 'image_node'].edge_index = torch.from_numpy(class_to_image_index) # convert to torch
        data['class_node'].pos = torch.stack(class_node_pos, dim=0) if class_node_pos else torch.empty((0, 2), dtype=torch.float32) # add class node positions for visualisation
        return data
    
    @property
    def name(self):
        return self.dataset._name

    @property
    def image_shape(self): # [CHW]
        return list(self.dataset._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.dataset.image_shape) == 3 # CHW
        return self.dataset.image_shape[0]

    @property
    def resolution(self):
        assert len(self.dataset.image_shape) == 3 # CHW
        assert self.dataset.image_shape[1] == self.dataset.image_shape[2]
        return self.dataset.image_shape[1]
    

# ----------------------------------------------------------------------------

# variant of COCOStuffGraph to load precomputes np files including graphs
class COCOStuffGraphPrecomputedDataset(GeoDataset):
    def __init__(self,
                path,                       # Path to directory or zip for images/masks/graphs.
                graph_transform = None,     # Transform to apply to the graph.
                max_size    = None,         # Maximum number of items to load.
                random_seed = 0,            # Random seed to use when applying max_size.
                cache       = True,    # Cache images in CPU memory?
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
        return len(self._data_fnames)
    
    def _load_coco_files(self, idx):
        raw_idx = self._raw_idx[idx]
        if self._cache and raw_idx in self._cached_idxs:
            return self._cached_images[raw_idx], self._cached_masks[raw_idx], self._cached_graphs[raw_idx]

        img = self._load_np_from_path(self._data_fnames[raw_idx] + self._suffixes[0])
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

        data.image = torch.from_numpy(img[np.newaxis,...]) # add image to data object
        data.mask_path = mask_path # add mask to data object

        # ---- IMAGE
        image_patch_placeholder = torch.empty((1,), dtype=torch.float32)
        data['image_node'].x = image_patch_placeholder
        data['image_node'].pos = self.image_patch_positions

        # ---- Class
        data['class_node'].x = torch.from_numpy(graphs['class_node']).to(torch.float32)
        data['class_node'].pos = self.safe_key_pos_open(graphs, 'class_pos')

        # ---- Edges
        data['class_node', 'class_edge', 'class_node'].edge_index = torch.from_numpy(graphs['class_edge']).to(torch.long) if graphs['class_edge'].shape[1] != 0 else torch.empty((2, 0), dtype=torch.long)
        data['class_node', 'class_to_image', 'image_node'].edge_index = torch.from_numpy(graphs['class_to_image']).to(torch.long) if graphs['class_to_image'].shape[1] != 0 else torch.empty((2, 0), dtype=torch.long)

        return data
    
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