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

import json
from collections import defaultdict
import numpy as np

"""
Class to open preprocessed vg annotations, center crop them, and pair them with training images
"""
class VGAnnoations():
    def __init__(self,
                 vg_annotation_path, # path to h5 dataset containing 
                 min_object_size=0.02,):
        
        


    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):

        for idx, obj_bbox in enumerate(bbox):
            if not is_valid_bbox[idx]:
                continue
            self.total_num_bbox += 1

            x, y, w, h = obj_bbox

            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 1, W)
            y1 = np.clip(y + h, 1, H)

            if (y1 - y0 < self.min_object_size) or (x1 - x0 < self.min_object_size):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def get_init_meta_data(self, image_id):
        # layout_length = self.max_objects_per_image + 2
        meta_data = {
            'filename': self.image_id_to_filename[image_id].replace('/', '_').split('.')[0]
        }

        return meta_data


    def __len__(self):
        return len(self.image_ids)
    

    import numpy as np

    def center_crop_bboxes_with_filter(self, orig_width: int, orig_height: int, bboxes: np.ndarray, min_area_ratio: float = 0.02):
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
    

    def save_vocab_to_json(self, json_file_path):
        # Save to JSON file
        with open(json_file_path, 'w') as f:
            json.dump(self.vocab, f, indent=4)

    def save_dataset_to_json(self, json_file_path):
        data_to_save = {}
        print(self.__len__())
        for i in range(self.__len__()):
            item = self.__getitem__(i)
            # Create a dictionary where the key is the filename and value is the rest of the data
            data_dict = {item['filename']: item}
            data_to_save.update(data_dict)
        
        print('files', len(data_to_save))
        # Save to JSON file
        with open(json_file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)


    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.

        """
        image_id = self.image_ids[index]
        

        W, H = self.image_id_to_size[image_id]
        num_obj = len(self.image_id_to_objects[image_id])
        obj_bbox = np.array([obj['bbox'] for obj in self.image_id_to_objects[image_id]])
        obj_class = np.array([obj['category_id'] for obj in self.image_id_to_objects[image_id]])
        is_valid_obj = [True for _ in range(num_obj)]

        # get meta data
        meta_data = self.get_init_meta_data(image_id=image_id)

        # filter invalid bbox

        if len(obj_bbox)>0:
            obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=H, W=W, bbox=obj_bbox, is_valid_bbox=is_valid_obj)
            obj_bbox = obj_bbox[is_valid_obj]
            obj_class = obj_class[is_valid_obj]

            obj_bbox, is_valid_bb = self.center_crop_bboxes_with_filter(W, H, obj_bbox)
            obj_class = obj_class[is_valid_bb]

        meta_data['obj_bbox'] = obj_bbox.tolist()
        meta_data['obj_class'] = obj_class.tolist()
        meta_data['obj_class_name'] = [self.vocab['object_idx_to_name'][int(class_id)] for class_id in meta_data['obj_class']]
        meta_data['caption'] = self.image_id_to_captions[image_id]

        return meta_data


