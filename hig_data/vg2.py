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
                 vocab_path,
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

        self.vocab = set()
        with h5py.File(vocab_path, 'r') as clip_vocab_file:
            self.vocab = set(clip_vocab_file['ids'].keys())
        print('loaded vocab')
        
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
        img_path = os.path.join(self.image_dir, str(img_id)+'.jpg')

        # Get the scene graph for the given index
        scene_graph = self.scene_graphs[img_id]
        scene_graph_objs = {k['object_id']:k for k in scene_graph['objects']}
        image_id = scene_graph['image_id']

        assert img_id == image_id, f"Image path ID {img_id} does not match scene graph image ID {image_id}"

        # Process objects
        objects = self.image_id_to_objects[image_id]

        # extract name, boxes, global_id_to_local_id
        filter_objs = [o for o in objects if self.clean_str(o['names'][0]) in self.vocab]
        obj_names = [self.clean_str(o['names'][0]) for o in filter_objs]
        boxes = [[o['x'], o['y'], o['x']+o['w'], o['y']+o['h']] for o in filter_objs]

        global_id_to_local_id = {o['object_id']: i for i, o in enumerate(filter_objs)}
        global_ids = [o['object_id'] for o in filter_objs]
        

        # Handle cropping and adjust bounding boxes
        image = torch.from_numpy(self._load_image_from_path(img_path))
        orig_h, orig_w = image.shape[-2:]
        image, boxes, valid_idxs = self.center_crop(image, boxes)

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
                        clean = self.clean_str(attr)
                        if clean in self.vocab:
                            attr_list.append((attr_idx, clean, idx)) # save as [attr_id, vocab_word, obj_id]
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
                clean_rel = self.clean_str(predicate)
                if clean_rel in self.vocab:
                    updated_triples.append((subj_local_id, clean_rel, obj_local_id)) # save as [obj_id, vocab_word, obj_id]


        return image, {'image_id': image_id,
                       'obj_class': obj_names,
                       'obj_bbox': boxes,
                       'triples': set(updated_triples),
                       'attributes': set(attr_list),
                       'orig_img_size': (orig_h, orig_w)}

    def clean_str(self, s):
        s = s.strip().lower()
        # keep letters, numbers, spaces, and basic punctuation: commas, periods, apostrophes, exclamation, question
        s = re.sub(r'[^a-z0-9 ,.\'!?]+', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def center_crop(self, img, obj_bbox, min_area=0.01):


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
            # Compute original area
            bw = (xmax - xmin) 
            bh = (ymax - ymin)
            original_area = bw * bh

            xmin_new = max(0, min(1, (xmin - left) / crop_size))
            ymin_new = max(0, min(1, (ymin - top) / crop_size))
            xmax_new = max(0, min(1, (xmax - left) / crop_size))
            ymax_new = max(0, min(1, (ymax - top) / crop_size))

            
            # Calculate new width and height after adjustment
            bw_new = (xmax_new - xmin_new)*self.image_size
            bh_new = (ymax_new - ymin_new)*self.image_size

            new_area = bw_new * bh_new

            # Check both minimum area and 50% reduction criteria
            if new_area >= min_area and new_area >= 0.5 * original_area:
                new_boxes.append([xmin_new, ymin_new, xmax_new, ymax_new])
                valid_idxs.append(idx)


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
                 path,                       # Path to file paths for h5 dataset and vocab
                 split='train',              # which split to select default is training split
                 transform=None,
                 max_size=None,
                 random_seed=0,
                 reverse_img_edges=False,
                 class_labels_var=0.3775,    # Calculated CLIP latent variance
                 **kwargs,
                 ):
        super().__init__(None, transform)

        with open(path, 'r') as f:
            paths = json.load(f) 

        self.path = paths['path'] # Path to the precomputed h5 file 
        self.vocab_path = paths['vocab_path'] # Path to the vocab h5 file
        self.split_path = paths['split_path'] # Path to dataset ID splits
        self.class_labels_var = class_labels_var

        self.reverse_img_edges = reverse_img_edges

        
        with h5py.File(self.path, 'r', libver='latest') as hdf:
            self.loadable_img_ids = set(hdf.keys())

        with open(self.split_path, 'r') as f:
            image_ids = list(json.load(f)[split])
            image_id_plus_flip = [str(i) for i in image_ids] + [f"{i}_f" for i in image_ids]
            self._data_fnames = sorted([i for i in image_id_plus_flip if i in self.loadable_img_ids])

        dataset_size = len(self._data_fnames)
        self._raw_idx = np.arange(dataset_size, dtype=np.int64)
        if (max_size is not None) and (len(self._raw_idx) > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        with h5py.File(self.path, 'r', libver='latest') as hdf:
            raw_ref_image = hdf[str(self._data_fnames[0])]['image'][:]

        self._raw_shape = [dataset_size] + list(raw_ref_image.shape)
        self.grid_size = self._raw_shape[-1] # grid size for image patches 

        self.vocab_lookup = {}
        self.vocab = []
        with h5py.File(self.vocab_path, 'r', libver='latest') as f:
            for i, (key, value) in enumerate(f['latents'].items()):
                self.vocab_lookup[key] = i
                self.vocab.append(torch.from_numpy(value[:]))
        self.vocab = torch.stack(self.vocab)

        self.latent_dim = 768
        self.image_patch_positions = get_image_patch_positions(image_size=self.grid_size)

    def __len__(self):
        return len(self._raw_idx)

    def _create_image_nodes(self, data, img):
        data.image = img.unsqueeze(0)
        image_patch_placeholder = torch.empty((data.image.shape[-2]*data.image.shape[-1], 1), dtype=torch.float32, device=data.image.device) # must init be same size for correct minibatching
        data['image_node'].x = image_patch_placeholder
        data['image_node'].pos = self.image_patch_positions
        return data

    def _create_attribute_nodes(self, data, attributes):
        if attributes.size > 0:
            attr_ids = attributes[:,0].astype(np.int64)
            attr_vocab = attributes[:,1].astype(np.int64)
            attr_obj_id = attributes[:,2].astype(np.int64)

            # create attribute nodes from clip vocab
            data['attribute_node'].x = torch.stack([self._get_vocab_latent(i) for i in attr_vocab]).to(torch.float32)
            data['attribute_node'].label = np.stack([i for i in attr_vocab])

            # create attribute to object edges, and reverse - otherwise they are simply isolated nodes
            edge_index = torch.stack([torch.from_numpy(attr_ids).long(), torch.from_numpy(attr_obj_id).long()])
            data['attribute_node', 'attr_to_instance', 'instance_node'].edge_index = edge_index
            reverse_attr = torch.stack([edge_index[1], edge_index[0]])
            data['instance_node', 'instance_to_attr', 'attribute_node'].edge_index = reverse_attr

        else: # cover no attributes present case
            data['attribute_node'].x = torch.empty((0, self.latent_dim), dtype=torch.float32)
            data['attribute_node'].label = torch.empty((0, 2), dtype=torch.long)
            data['attribute_node', 'attr_to_instance', 'instance_node'].edge_index = torch.empty((2,0), dtype=torch.long)
            data['instance_node', 'instance_to_attr', 'attribute_node'].edge_index = torch.empty((2,0), dtype=torch.long)
        return data

    def _create_relationship_edges(self, data, relationships):
        if relationships.size > 0:
            rel_obj1 = relationships[:,0].astype(np.int64)
            rel_vocab = relationships[:,1].astype(np.int64)
            rel_obj2 = relationships[:,2].astype(np.int64)

            edge_index = torch.stack([torch.from_numpy(rel_obj1).long(),
                                      torch.from_numpy(rel_obj2).long()])
            
            # edge_attr from clip vocab
            edge_attr = torch.stack([self._get_vocab_latent(i) for i in rel_vocab]).to(torch.float32)

            data['instance_node', 'rel_to_instance', 'instance_node'].edge_index = edge_index
            data['instance_node', 'rel_to_instance', 'instance_node'].edge_attr = edge_attr

            if self.reverse_img_edges:
                reverse_rel = torch.stack([edge_index[1], edge_index[0]])
                data['instance_node', 'instance_to_rel', 'instance_node'].edge_index = reverse_rel
                data['instance_node', 'instance_to_rel', 'instance_node'].edge_attr = edge_attr
        else: # cover empty case
            data['instance_node', 'rel_to_instance', 'instance_node'].edge_index = torch.empty((2,0), dtype=torch.long)
            data['instance_node', 'rel_to_instance', 'instance_node'].edge_attr = torch.empty((0, self.latent_dim), dtype=torch.float32)
        return data

    def _create_instance_nodes(self, data, obj_class, obj_bbox):
        if len(obj_class) == 0:
            data['instance_node'].x = torch.empty((0, self.latent_dim), dtype=torch.float32)
            data['instance_node'].pos = torch.empty((0, 2), dtype=torch.float32)
            data['instance_node'].label = torch.empty((0, 2), dtype=torch.long)
        else:
            obj_vocab = torch.stack([self._get_vocab_latent(i) for i in obj_class]).to(torch.float32)
            data['instance_node'].label = np.stack([i for i in obj_class])
            data['instance_node'].x = obj_vocab
            pos = np.array([((x_min+x_max)/2, (y_min+y_max)/2) for x_min,y_min,x_max,y_max in obj_bbox])
            data['instance_node'].pos = torch.from_numpy(pos).to(torch.float32)
        return data
    
    # def __del__(self):
    #     # Close the HDF5 file when the dataset is deleted
    #     if self.vocab_file is not None:
    #         self.vocab_file.close()
    
    def _get_vocab_latent(self, class_idx):
        default = torch.zeros((self.latent_dim), dtype=torch.float32)
        if str(class_idx) in self.vocab_lookup:
            index = self.vocab_lookup[str(class_idx)]
            latent = self.vocab[index]
            if torch.isnan(latent).any():
                return default
            return self.vocab[index] / np.sqrt(self.class_labels_var)
        return default

    def _create_instance_to_image_edges(self, data, obj_bbox,):

        # from bounding boxes convert to linear image idxs
        instance_to_image_edges = []
        for i, bbox in enumerate(obj_bbox):
            linear_indices = bbox_to_linear_indices(bbox, image_size=(self.grid_size, self.grid_size))
            if len(linear_indices) > 0:
                node_id_repeated = np.full((len(linear_indices),), i, dtype=np.int64)
                edge_index = np.stack([node_id_repeated, linear_indices], axis=0)
                instance_to_image_edges.append(edge_index)

        if len(instance_to_image_edges) > 0:
            instance_to_image_index = np.concatenate(instance_to_image_edges, axis=1)
        else:
            instance_to_image_index = np.zeros((2,0), dtype=np.int64)

        # create edges in hetero obj
        data['instance_node', 'instance_to_image', 'image_node'].edge_index = torch.from_numpy(instance_to_image_index).to(torch.long)
        if self.reverse_img_edges:
            reverse_edge_index = torch.stack([data['instance_node', 'instance_to_image', 'image_node'].edge_index[1],
                                              data['instance_node', 'instance_to_image', 'image_node'].edge_index[0]])
            data['image_node', 'image_to_instance', 'instance_node'].edge_index = reverse_edge_index
        return data

    def __getitem__(self, idx: int) -> HeteroData:

        data = RelaxedHeteroData()
        raw_idx = self._raw_idx[idx]
        fname = str(self._data_fnames[raw_idx])

        with h5py.File(self.path, 'r',) as hdf:
            group = hdf[fname]
            img = torch.from_numpy(group['image'][:]).float()
            obj_class = group['obj_class'][:].astype(np.int64)
            obj_bbox = group['obj_bbox'][:].astype(np.float32)
            print(obj_bbox)
            relationships = group['relationships'][:]
            attributes = group['attributes'][:]

        # Create the various nodes and edges
        data = self._create_image_nodes(data, img)
        data = self._create_instance_nodes(data, obj_class, obj_bbox)
        data = self._create_attribute_nodes(data, attributes)
        data = self._create_relationship_edges(data, relationships)
        data = self._create_instance_to_image_edges(data, obj_bbox,)

        data.caption = torch.mean(data['instance_node'].x, dim=0, keepdim=True) # take average of clip latents for instances in image
        if torch.isnan(data.caption).any():
            data.caption = torch.zeros((1, self.latent_dim), dtype=torch.float32)

        data.fname = fname
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




class H5LatentReader:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.file = None

    def __enter__(self):
        self.file = h5py.File(self.vocab_path, 'r', libver='latest', swmr=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()

    def get_vocab_latent(self, class_idx):
        if self.file is None:
            raise ValueError("File not opened. Use the context manager or explicitly open it.")
        group = self.file.get('latents')
        if group and str(class_idx) in group:
            return torch.from_numpy(group[str(class_idx)][:]).to(torch.float32)
        return torch.zeros((self.latent_dim), dtype=torch.float32)
