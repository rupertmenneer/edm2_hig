#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from collections import defaultdict
import torch
import numpy as np
import h5py
import PIL
import json

class VgSceneGraphDataset(torch.utils.data.Dataset):

    def __init__(self,
                vocab_json,
                h5_path,
                image_dir,
                image_size=(256, 256),
                max_samples=None,
                include_relationships=True,
                use_orphaned_objects=True,
                min_object_size=0.01
        ):

        super(VgSceneGraphDataset, self).__init__()

        self.image_dir = image_dir
        self.image_size = image_size
        self.vocab_json = vocab_json
        with open(vocab_json, 'r') as f:
            self.vocab = json.load(f)  
        self.num_objects = len(self.vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        self.min_object_size=min_object_size

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
            (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
            means that (objs[i], p, objs[j]) is a triple.
        """
        raw_path = self.image_paths[index].decode('utf-8')
        raw_base_name = os.path.splitext(os.path.basename(raw_path))[0]
        img_path = os.path.join(self.image_dir, os.path.basename(raw_base_name)+'.png')
        WW, HH = np.array(self.data['image_widths'][index]), np.array(self.data['image_heights'][index])
        obj_idxs = np.array(range(self.data['objects_per_image'][index].item()))
        n_objs = len(obj_idxs) 
        is_valid_obj = [True for _ in range(n_objs)]
        obj_bbox = np.array(self.data['object_boxes'][index][:n_objs])
        obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=HH, W=WW, bbox=obj_bbox, is_valid_bbox=is_valid_obj)
        obj_bbox = obj_bbox[is_valid_obj]
        obj_bbox, is_valid_bb = self.center_crop_bboxes_with_filter(WW, HH, bboxes=obj_bbox) # crop and resize w.r.t to original image size

        # create mapping
        obj_idxs = obj_idxs[is_valid_bb]
        n_objs = len(obj_idxs) 
        objs = torch.LongTensor(n_objs).fill_(-1)
        obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            obj_idx_mapping[obj_idx] = i

        triples = []
        for r_idx in range(self.data['relationships_per_image'][index].item()):
            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        triples = torch.LongTensor(triples)
        return objs, obj_bbox, triples, img_path


    def center_crop_bboxes_with_filter(self, orig_width: int, orig_height: int, bboxes: np.ndarray, min_area_ratio: float = 0.01):
        """
        Adjust bounding boxes in pixel coordinates for a center crop. Discards boxes outside the image or below a minimum size.
        
        Arguments:
            orig_width: The original width of the image.
            orig_height: The original height of the image.
            bboxes: Bounding boxes in pixel space, as an Nx4 numpy array where each row is [xmin, ymin, xmax, ymax].
            min_area_ratio: The minimum area of bounding box as a ratio of the image size to retain the box (default is 2%).
        
        Returns:
            Adjusted bounding boxes in 0-1 range after cropping, agnostic of image size.
        """

        # Calculate scale factor for center cropping (target crop is square)
        crop_size = min(orig_width, orig_height)
        crop_x_offset = (orig_width - crop_size) // 2
        crop_y_offset = (orig_height - crop_size) // 2
        
        # Adjust the bounding boxes by subtracting the cropping offsets
        bboxes[:, [0, 2]] -= crop_x_offset
        bboxes[:, [1, 3]] -= crop_y_offset
        
        # Clip bounding boxes to be within the cropped area
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, crop_size)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, crop_size)
        
        # Calculate the area of the bounding boxes
        box_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        
        # Compute minimum area threshold (in pixels)
        min_area = min_area_ratio * (crop_size * crop_size)
        
        # Filter out boxes that are too small or fully outside the image
        valid_boxes = (box_areas >= min_area) & ((bboxes[:, 2] - bboxes[:, 0]) > 0) & ((bboxes[:, 3] - bboxes[:, 1]) > 0)
        
        # Get the indices of valid bounding boxes
        valid_idxs = np.where(valid_boxes)[0]
        
        # Filter bounding boxes based on validity
        bboxes = bboxes[valid_boxes]
        
        # Convert bounding boxes to normalized 0-1 range
        bboxes_normalized = bboxes / crop_size
        
        return bboxes_normalized, valid_idxs
    
    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):

        for idx in range(len(is_valid_bbox)):
            
            if not is_valid_bbox[idx]:
                continue

            obj_bbox = bbox[idx]
            x, y, w, h = obj_bbox

            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 1, W)
            y1 = np.clip(y + h, 1, H)

            if (y1 - y0 < self.min_object_size) or (x1 - x0 < self.min_object_size):
                is_valid_bbox[idx] = False
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox