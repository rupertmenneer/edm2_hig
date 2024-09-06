import torchvision
import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset as GeoDataset

from graph_datasets.hf_dataset import HuggingFaceDataset

"""
Loads COCO dataset from HF.
"""
# RUPERT MENNEER (2024)
#----------------------------------------------------------------------------

class COCOStuffDataset(HuggingFaceDataset):

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
        graph_transform=None,
        patch_size = 8,
        n_labels = 182,
    ) -> None:

        super().__init__(None, graph_transform)
        self.dataset = COCOStuffDataset()
        self.n_labels = n_labels

        # image node positions (same for every img)
        self.grid_size = self.dataset._target_resolution[0] // patch_size
        self.num_image_nodes = self.grid_size * self.grid_size
        

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

        # create class nodes from semantic segmentation map
        self._create_class_nodes(data, mask)

        # create class to image edges from semantic segmentation map
        self._create_class_to_image_edges(data, mask)

        if self.transform: # apply any additional transforms
            data = self.transform(data)

        return data
    
    def _create_class_nodes(self, data, mask):
        class_labels = np.array([l for l in np.unique(mask) if l != 255], dtype=np.int16)
        onehots = np.zeros((len(class_labels), self.n_labels), dtype=np.float32)
        onehots[np.arange(len(class_labels)), class_labels] = 1
        onehots = np.concatenate([class_labels[...,np.newaxis], onehots], axis=1) # add class labels to onehots position 0 for convenience
        data['class_node'].x = torch.from_numpy(onehots).to(torch.float32)
        # densely connect class nodes
        edge_index = torch.combinations(torch.arange(6), with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data['class_node', 'class_edge', 'class_node'].edge_index = edge_index
        return data
    
    def _create_class_to_image_edges(self, data, mask):
        edges = []
        resized_mask = torch.nn.functional.interpolate(torch.tensor(mask[np.newaxis,...], dtype=torch.float32), size=(self.grid_size, self.grid_size), mode='nearest').squeeze() # resize mask to match compression
        for class_node_idx, class_label in enumerate(data['class_node'].x[:,0]):
            class_mask = np.argwhere(resized_mask == class_label) # get mask idxs for current class
            linear_image_patch_print_line_idxs = (class_mask[0] * self.grid_size + class_mask[1]).long() # linearise image node idxs
            node_id_repeated = np.full((len(linear_image_patch_print_line_idxs),), class_node_idx, dtype=np.int16) # repeat class node idx for each patch
            edge_index = np.stack([node_id_repeated, linear_image_patch_print_line_idxs], axis=0)
            edges.append(edge_index)
        class_to_image_index = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int16)
        data['class_node', 'class_to_image', 'image_node'].edge_index = torch.from_numpy(class_to_image_index) # convert to torch
        return data
    

# ----------------------------------------------------------------------------
import re
class RelaxedHeteroData(HeteroData):
    # modified to allow allow_empty flag to be passed
    def __getattr__(self, key: str, allow_empty=True):
        # `data.*_dict` => Link to node and edge stores.
        # `data.*` => Link to the `_global_store`.
        # Using `data.*_dict` is the same as using `collect()` for collecting
        # nodes and edges features.
        if hasattr(self._global_store, key):
            return getattr(self._global_store, key)
        elif bool(re.search('_dict$', key)):
            return self.collect(key[:-5], allow_empty=allow_empty)
        raise AttributeError(f"'{self.__class__.__name__}' has no "
                             f"attribute '{key}'")