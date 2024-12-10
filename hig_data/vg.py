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
from hig_data.coco2 import Dataset
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as T

class VgSceneGraphDataset(torch.utils.data.Dataset):

    def __init__(self,
                vocab_json,
                h5_path,
                image_dir,
                image_size=512,
                max_samples=None,
                include_relationships=True,
                use_orphaned_objects=True,
                min_object_size=0.01
        ):

        super(VgSceneGraphDataset, self).__init__()

        self.image_dir = image_dir
        self.size = image_size
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
        img_path = os.path.join(self.image_dir, os.path.basename(raw_base_name)+'.jpg')
        HH, WW = np.array(self.data['image_widths'][index]), np.array(self.data['image_heights'][index]) # wrong way round
        obj_idxs = np.array(range(self.data['objects_per_image'][index].item()))
        n_objs = len(obj_idxs) 

        obj_bbox = np.array(self.data['object_boxes'][index][:n_objs])
        img, obj_bbox, is_valid_obj = self.center_crop(img_path, obj_bbox)
        
        # create mapping
        obj_idxs = obj_idxs[is_valid_obj]
        n_objs = len(obj_idxs) 
        obj_idx_mapping = {}
        objs=[]
        for i, obj_idx in enumerate(obj_idxs):
            objs.append(self.data['object_names'][index, obj_idx].item())
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

        
        obj_attrs = []
        object_ids = self.data['object_ids'][index]

        for i, obj_idx in enumerate(obj_idxs):
            raw_obj_idx = object_ids[obj_idx]
            for a_idx in range(self.data['attributes_per_object'][index, obj_idx].item()):
                object_attribute = self.data['object_attributes'][raw_obj_idx, a_idx]
                obj_attrs.append([obj_idx, object_attribute])

        triples = torch.LongTensor(triples)
        return objs, obj_bbox, triples, img_path, img, attributes_per_object, object_attributes
    
    def center_crop(self, img, obj_bbox):

        # img = PIL.Image.open(img)
        img = torch.from_numpy(self._load_image_from_path(img))

        # Calculate center crop parameters
        width, height = img.shape[-1], img.shape[-2]

        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2

        # Perform the center crop
        img = img[:, top:top + crop_size, left:left + crop_size]

        # Resize to the target size
        img = F.resize(img, (self.size, self.size), interpolation=T.InterpolationMode.BICUBIC)

        # Threshold for minimum bounding box area (2% of cropped image area)
        min_area = 0.02 * self.size * self.size
        new_boxes = []
        valid_idxs = []

        # Adjust bounding boxes based on crop
        for idx, bbox in enumerate(obj_bbox):

            xmin, ymin, width, height = bbox
            xmax, ymax = xmin + width, ymin + height

            xmin_new = max(0, min(1, (xmin - left) / crop_size))
            ymin_new = max(0, min(1, (ymin - top) / crop_size))
            xmax_new = max(0, min(1, (xmax - left) / crop_size))
            ymax_new = max(0, min(1, (ymax - top) / crop_size))

            
            # Calculate new width and height after adjustment
            bw_new = (xmax_new - xmin_new)*self.size
            bh_new = (ymax_new - ymin_new)*self.size

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


    # def center_crop_bboxes_with_filter(self, orig_width: int, orig_height: int, bboxes: np.ndarray, min_area: float):
    #     # Calculate center crop parameters
    #     crop_size = min(orig_width, orig_height)
    #     left = (orig_width - crop_size) // 2
    #     top = (orig_height - crop_size) // 2

    #     # Adjust `min_area` to scale with the new size
    #     min_area = min_area * self.size * self.size

    #     new_boxes = []
    #     valid_idxs = []

    #     # Adjust bounding boxes based on crop
    #     for idx, bbox in enumerate(bboxes):
    #         xmin, ymin, width, height = bbox
    #         xmax, ymax = xmin + width, ymin + height

    #         # Adjust box coordinates relative to the cropped region
    #         xmin_new = (xmin - left) / crop_size * self.size
    #         ymin_new = (ymin - top) / crop_size * self.size
    #         xmax_new = (xmax - left) / crop_size * self.size
    #         ymax_new = (ymax - top) / crop_size * self.size

    #         # Clamp to ensure coordinates are within [0, self.size]
    #         xmin_new = max(0, min(self.size, xmin_new))
    #         ymin_new = max(0, min(self.size, ymin_new))
    #         xmax_new = max(0, min(self.size, xmax_new))
    #         ymax_new = max(0, min(self.size, ymax_new))

    #         # Calculate new width and height after adjustment
    #         bw_new = xmax_new - xmin_new
    #         bh_new = ymax_new - ymin_new

    #         # Check if the new box area meets the minimum threshold
    #         if bw_new > 0 and bh_new > 0 and bw_new * bh_new >= min_area:
    #             new_boxes.append([xmin_new, ymin_new, xmax_new, ymax_new])
    #             valid_idxs.append(idx)  # Store the original index of valid boxes

    #     return np.array(new_boxes), valid_idxs


    # def center_crop_bboxes_with_filter(self, orig_width: int, orig_height: int, bboxes: np.ndarray, min_area_ratio: float = 0.01):
    #     """
    #     Adjust bounding boxes in pixel coordinates for a center crop. Discards boxes outside the image or below a minimum size.
        
    #     Arguments:
    #         orig_width: The original width of the image.
    #         orig_height: The original height of the image.
    #         bboxes: Bounding boxes in pixel space, as an Nx4 numpy array where each row is [xmin, ymin, xmax, ymax].
    #         min_area_ratio: The minimum area of bounding box as a ratio of the image size to retain the box (default is 1%).
        
    #     Returns:
    #         Adjusted bounding boxes in 0-1 range after cropping, agnostic of image size.
    #     """
    #     # Calculate scale factor for center cropping (target crop is square)
    #     crop_size = min(orig_width, orig_height)
    #     crop_x_offset = (orig_width - crop_size) // 2
    #     crop_y_offset = (orig_height - crop_size) // 2
        
    #     # Adjust the bounding boxes by subtracting the cropping offsets
    #     bboxes[:, [0, 2]] -= crop_x_offset
    #     bboxes[:, [1, 3]] -= crop_y_offset
        
    #     # Clip bounding boxes to be within the cropped area
    #     bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, crop_size)
    #     bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, crop_size)
        
    #     # Calculate the width and height of each bounding box after clipping
    #     box_widths = bboxes[:, 2] - bboxes[:, 0]
    #     box_heights = bboxes[:, 3] - bboxes[:, 1]
        
    #     # Filter out boxes that are fully outside the image
    #     valid_boxes = (box_widths > 0) & (box_heights > 0)
    #     bboxes = bboxes[valid_boxes]
    #     box_widths = box_widths[valid_boxes]
    #     box_heights = box_heights[valid_boxes]
        
    #     # Calculate the area of the bounding boxes
    #     box_areas = box_widths * box_heights


    #     # Compute minimum area threshold (in pixels)
    #     # min_area = min_area_ratio * (crop_size * crop_size)
    #     print(box_areas, min_area)
        
    #     # Further filter boxes by area
    #     # valid_area_boxes = box_areas >= min_area
    #     # bboxes = bboxes[valid_area_boxes]
        
    #     # Convert bounding boxes to normalized 0-1 range
    #     bboxes_normalized = bboxes / crop_size
        
    #     # Return valid indices and normalized bounding boxes
    #     # valid_idxs = np.where(valid_boxes)[0][valid_area_boxes]
    #     return bboxes_normalized, valid_idxs
    
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

            # if (w*h < self.min_object_size):
                # is_valid_bbox[idx] = False
                # continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox
    


