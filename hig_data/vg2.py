import os
from collections import defaultdict
import torch
import numpy as np
import PIL
import ujson as json
import torchvision.transforms as T
import torchvision
import torchvision.transforms.functional as F


class VgSceneGraphDataset(torch.utils.data.Dataset):
    def __init__(self,
                 scene_graph_json,
                 objects_json,
                 image_dir,
                 image_size=512,
                 ):
        super(VgSceneGraphDataset, self).__init__()
        self.image_dir = image_dir
        self.image_size = image_size

        # Load scene graphs from JSON
        with open(scene_graph_json, 'r') as f:
            scene_graphs = json.load(f)
        print('loaded scene graph json')
        self.scene_graphs = {}
        for s in scene_graphs:
            self.scene_graphs[s['image_id']] = s
        with open(objects_json, 'r') as f:
            objects = json.load(f)
        print('loaded obj json')
        self.image_id_to_objects = {}
        for image in objects:
            image_id = image['image_id']
            self.image_id_to_objects[image_id] = image['objects']
        

        supported_ext = {'.jpg', '.jpeg', '.png', '.npy'}
        # self.image_fnames = sorted(fname for fname in set(self.get_filelist_from_dir(image_dir)) if self._file_ext(fname) in supported_ext)
        self.image_ids = list(self.scene_graphs.keys())
        print('sorted filenames')



    def get_filelist_from_dir(self, path):
        return {os.path.relpath(os.path.join(root, fname), start=path) for root, _dirs, files in os.walk(path) for fname in files}

    def __len__(self):
        return len(self.scene_graphs)

    def __getitem__(self, index,):

        img_id = self.image_ids[index]
        # image_path_id = int(os.path.splitext(img_fname)[0])
        img_path = os.path.join(self.image_dir, str(img_id)+'.jpg')

        # Get the scene graph for the given index
        scene_graph = self.scene_graphs[img_id]
        scene_graph_objs = {k['object_id']:k for k in scene_graph['objects']}
        image_id = scene_graph['image_id']

        assert img_id == image_id, f"Image path ID {img_id} does not match scene graph image ID {image_id}"

        # Process objects
        # objects = scene_graph['objects']
        objects = self.image_id_to_objects[image_id]

        # extract name, boxes, global_id_to_local_id
        obj_names = [self.clean_str(', '.join(o['names'])) if 'names' in o else 'object' for o in objects]
        boxes = [[o['x'], o['y'], o['x']+o['w'], o['y']+o['h']] for o in objects]
        global_id_to_local_id = {o['object_id']: i for i, o in enumerate(objects)}
        global_ids = [o['object_id'] for o in objects]
        

        # Handle cropping and adjust bounding boxes
        image, boxes, valid_idxs = self.center_crop(img_path, boxes)

        # Update objs and global_ids based on valid indices
        obj_names = [obj_names[i] for i in valid_idxs]
        global_ids = [global_ids[i] for i in valid_idxs]
        # update mapping from global object IDs to new local IDs after cropping
        global_id_to_local_id = {global_id: idx for idx, global_id in enumerate(global_ids)}

        # Process attributes for valid objects
        attr_list = []
        attr_idx = 0
        for idx, obj_idx in enumerate(global_ids):
            obj = scene_graph_objs.get(obj_idx, None)
            if obj and 'attributes' in obj:
                attrs = obj['attributes']
                for attr in attrs:
                    if attr:
                        attr_list.append((attr_idx, self.clean_str(attr), idx)) # save as [attr_id, vocab_word, obj_id]
                        attr_idx += 1

        # Process relationships involving valid objects
        relationships = scene_graph['relationships']
        updated_triples = []
        for rel in relationships:
            subject_id = rel['subject_id']
            object_id = rel['object_id']
            predicate = rel['predicate']

            subj_local_id = global_id_to_local_id.get(subject_id)
            obj_local_id = global_id_to_local_id.get(object_id)

            if subj_local_id is not None and obj_local_id is not None and predicate:
                updated_triples.append((subj_local_id, self.clean_str(predicate), obj_local_id)) # save as [obj_id, vocab_word, obj_id]


        return image, {'image_id': image_id, 'obj_class': obj_names, 'obj_bbox': boxes, 'triples': set(updated_triples), 'attributes': set(attr_list),}

    def clean_str(self, s):
        return s.lower().replace('/', '_')

    def center_crop(self, img_path, obj_bbox, min_area=0.01):

        img = torch.from_numpy(self._load_image_from_path(img_path))

        # Calculate center crop parameters
        width, height = img.shape[-1], img.shape[-2]

        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2

        # Perform the center crop
        img = img[:, top:top + crop_size, left:left + crop_size]

        # Resize to the target size
        img = F.resize(img, (self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC)

        # Threshold for minimum bounding box area (2% of cropped image area)
        min_area = min_area * self.image_size * self.image_size
        new_boxes = []
        valid_idxs = []

        # Adjust bounding boxes based on crop
        for idx, bbox in enumerate(obj_bbox):

            xmin, ymin, xmax, ymax = bbox

            xmin_new = max(0, min(1, (xmin - left) / crop_size))
            ymin_new = max(0, min(1, (ymin - top) / crop_size))
            xmax_new = max(0, min(1, (xmax - left) / crop_size))
            ymax_new = max(0, min(1, (ymax - top) / crop_size))

            
            # Calculate new width and height after adjustment
            bw_new = (xmax_new - xmin_new)*self.image_size
            bh_new = (ymax_new - ymin_new)*self.image_size

            # Check if the new box area meets the minimum threshold
            if bw_new * bh_new >= min_area:
                new_boxes.append([xmin_new, ymin_new, xmax_new, ymax_new])
                valid_idxs.append(idx)  # Store the original index of valid boxes


        return img, new_boxes, valid_idxs

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def _load_image_from_path(self, fname): # return image from fname as np.ndarray
        ext = self._file_ext(fname)
        with open(fname, 'rb') as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f).convert("RGB"))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image


# ----------------------------------------------------------------------------
from torch_geometric.data import HeteroData, Dataset as GeoDataset
import h5py
import os
import json
import numpy as np
import torch


class VGGraphPrecomputedDataset(GeoDataset):
    def __init__(self,
                 path,                       # Path to the precomputed h5 file 
                 vocab_path,                 # Path to the vocab h5 file
                 split_path,                 # Path to dataset ID splits
                 split='train',              # which split to select default is training split
                 transform=None,
                 max_size=None,
                 random_seed=0,
                 reverse_img_edges=False):
        super().__init__(None, transform)

        self.path = path
        self.vocab_path = vocab_path
        self.reverse_img_edges = reverse_img_edges
        self._cache = False

        with open(split_path, 'r') as f:
            self._data_fnames = sorted(json.load(f)[split])

        # with h5py.File(self.path, 'r') as hdf:
        #     # no 'ids' dataset now, just groups per image id
        #     self._data_fnames = sorted(list(hdf.keys()))

        dataset_size = len(self._data_fnames)
        self._raw_idx = np.arange(dataset_size, dtype=np.int64)
        if (max_size is not None) and (len(self._raw_idx) > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Load vocab file 
        with h5py.File(self.vocab_path, 'r') as f:
            # self.vocab = f['latents']
            self.vocab = {k: v[:] for k, v in f['latents'].items()}

    def __len__(self):
        return len(self._raw_idx)

    def _create_image_nodes(self, data, img):
        C, H, W = img.shape
        data.image = img.unsqueeze(0)
        image_patches = img.permute(1,2,0).reshape(H*W, C) # [H*W, C]

        data['image_node'].x = image_patches
        assert H==W, 'width and height should match, only square aspect ratio supported'
        data['image_node'].pos = get_image_patch_positions(image_size=H)
        return H, W

    def _create_attribute_nodes(self, data, attributes):
        if attributes.size > 0:
            attr_ids = attributes[:,0].astype(np.int64)
            attr_vocab = attributes[:,1].astype(np.int64)
            attr_obj_id = attributes[:,2].astype(np.int64)

            # create attribute nodes from clip vocab
            attr_vocab = np.stack([self._get_vocab_latent(i) for i in attr_vocab])
            data['attribute_node'].x = torch.from_numpy(attr_vocab).to(torch.float32)

            # create attribute to object edges
            edge_index = torch.stack([torch.from_numpy(attr_ids).long(),
                            torch.from_numpy(attr_obj_id).long()])
            data['attribute_node', 'attr_to_instance', 'instance_node'].edge_index = edge_index


            reverse_attr = torch.stack([edge_index[1], edge_index[0]])
            data['instance_node', 'instance_to_attr', 'attribute_node'].edge_index = reverse_attr

        else: # cover empty case
            data['attribute_node'].x = torch.empty((0,), dtype=torch.long)
            data['attribute_node'].attr_id = torch.empty((0,), dtype=torch.long)
            data['attribute_node', 'attr_to_instance', 'instance_node'].edge_index = torch.empty((2,0), dtype=torch.long)
        return data

    def _create_relationship_edges(self, data, relationships):
        if relationships.size > 0:
            rel_obj1 = relationships[:,0].astype(np.int64)
            rel_vocab = relationships[:,1].astype(np.int64)
            rel_obj2 = relationships[:,2].astype(np.int64)

            edge_index = torch.stack([torch.from_numpy(rel_obj1).long(),
                                      torch.from_numpy(rel_obj2).long()])
            
            # edge_attr from clip vocab
            edge_attr = torch.tensor([self._get_vocab_latent(i) for i in rel_vocab]).to(torch.float32)

            data['instance_node', 'rel_to_instance', 'instance_node'].edge_index = edge_index
            data['instance_node', 'rel_to_instance', 'instance_node'].edge_attr = edge_attr

            if self.reverse_img_edges:
                reverse_rel = torch.stack([edge_index[1], edge_index[0]])
                data['instance_node', 'instance_to_rel', 'instance_node'].edge_index = reverse_rel
                data['instance_node', 'instance_to_rel', 'instance_node'].edge_attr = edge_attr
        else: # cover empty case
            data['instance_node', 'rel_to_instance', 'instance_node'].edge_index = torch.empty((2,0), dtype=torch.long)
        return data

    def _create_instance_nodes(self, data, obj_class):
        obj_vocab = np.stack([self._get_vocab_latent(i) for i in obj_class])
        data['instance_node'].x = torch.from_numpy(obj_vocab).to(torch.float32)
        return data
    
    def _get_vocab_latent(self, class_idx):
        label = self.vocab[str(class_idx)][:] # lookup with str idx to get the vector describing the vocab clip latent
        return label

    def _create_instance_to_image_edges(self, data, obj_bbox, H, W):

        # from bounding boxes convert to linear image idxs
        instance_to_image_edges = []
        for i, bbox in enumerate(obj_bbox):
            linear_indices = bbox_to_linear_indices(bbox, (H,W))
            if len(linear_indices) > 0:
                node_id_repeated = np.full((len(linear_indices),), i, dtype=np.int64)
                edge_index = np.stack([node_id_repeated, linear_indices], axis=0)
                instance_to_image_edges.append(edge_index)

        if len(instance_to_image_edges) > 0:
            instance_to_image_index = np.concatenate(instance_to_image_edges, axis=1)
        else:
            instance_to_image_index = np.zeros((2,0), dtype=np.int64)

        # create edges in hetero obj
        data['instance_node', 'instance_to_image', 'image_node'].edge_index = torch.from_numpy(instance_to_image_index)
        if self.reverse_img_edges:
            reverse_edge_index = torch.stack([data['instance_node', 'instance_to_image', 'image_node'].edge_index[1],
                                              data['instance_node', 'instance_to_image', 'image_node'].edge_index[0]])
            data['image_node', 'image_to_instance', 'instance_node'].edge_index = reverse_edge_index
        return data

    def __getitem__(self, idx: int) -> HeteroData:
        data = RelaxedHeteroData()
        raw_idx = self._raw_idx[idx]
        fname = str(self._data_fnames[raw_idx])

        with h5py.File(self.path, 'r') as hdf:
            group = hdf[fname]
            img = torch.from_numpy(group['image'][:]).float()
            obj_class = group['obj_class'][:].astype(np.int64)
            obj_bbox = group['obj_bbox'][:].astype(np.float32)
            relationships = group['relationships'][:]
            attributes = group['attributes'][:]

        # Create the various nodes and edges
        H, W = self._create_image_nodes(data, img)
        data = self._create_instance_nodes(data, obj_class)
        data = self._create_attribute_nodes(data, attributes)
        data = self._create_relationship_edges(data, relationships)
        data = self._create_instance_to_image_edges(data, obj_bbox, H, W)

        return data
        
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
        assert len(self.image_shape) == 3 or len(self.image_shape) == 4 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]
    
    def __getstate__(self):
        return dict(self.__dict__,)


# ----------------------------------------------------------------------------

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

def linear_to_avg_2d_idx(idx, img_width=64, patch_size=8):
    row = torch.mean((idx // img_width).float()) * patch_size
    col = torch.mean((idx % img_width).float()) * patch_size
    return torch.tensor([col, row])

def linear_to_2d_idx(idx, img_width=64, patch_size=1):
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