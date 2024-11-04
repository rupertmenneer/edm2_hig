import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch

class HIGAugmentation(object):
    def __init__(self, size=512):
        self.size = size

        # Spatial augmentations that affect image, mask, and bounding boxes
        self.spatial_augs = [
            self.random_resized_crop,
            self.random_horizontal_flip
        ]

        # Color augmentations that only affect the image
        self.augment = T.Compose([
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)
        ])

    def __call__(self, img, mask, labels):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Apply spatial augmentations to img, mask, and bounding boxes
        for aug in self.spatial_augs:
            img, mask, labels = aug(img, mask, labels)
        
        # Apply color augmentations to the image only
        img = self.augment(img)

        return img, mask, labels

    def random_resized_crop(self, img, mask, labels):
        # Random resized crop with the specified size
        i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(0.5, 1.5), ratio=(1., 1.))
        
        img = F.resized_crop(img, i, j, h, w, (self.size, self.size), interpolation=T.InterpolationMode.BICUBIC)
        mask = F.resized_crop(mask, i, j, h, w, (self.size, self.size), interpolation=T.InterpolationMode.NEAREST)

        # Threshold for minimum bounding box area (2% of cropped image area)
        min_area = 0.02 * self.size * self.size
        new_boxes = []
        valid_idxs = []

            # Adjust bounding boxes based on crop
        for idx, bbox in enumerate(labels['obj_bbox']):
            xmin, ymin, xmax, ymax = bbox

            # Adjust box coordinates based on crop
            xmin_new = ((xmin - j) / w) * self.size
            ymin_new = ((ymin - i) / h) * self.size
            xmax_new = ((xmax - j) / w) * self.size
            ymax_new = ((ymax - i) / h) * self.size

            # Clamp to ensure coordinates are within [0, self.size]
            xmin_new = max(0, min(self.size, xmin_new))
            ymin_new = max(0, min(self.size, ymin_new))
            xmax_new = max(0, min(self.size, xmax_new))
            ymax_new = max(0, min(self.size, ymax_new))
            
            # Calculate new width and height after adjustment
            bw_new = xmax_new - xmin_new
            bh_new = ymax_new - ymin_new

            # Check if the new box area meets the minimum threshold
            if bw_new * bh_new >= min_area:
                new_boxes.append([xmin_new, ymin_new, xmax_new, ymax_new])
                valid_idxs.append(idx)  # Store the original index of valid boxes

        # Filter labels using valid_idxs
        for key in ['obj_bbox', 'obj_class', 'obj_class_name']:
            labels[key] = [labels[key][idx] for idx in valid_idxs]
        labels['obj_bbox'] = new_boxes  # Update with new bounding boxes

        return img, mask, labels


    def random_horizontal_flip(self, img, mask, labels):
        # Random horizontal flip with 50% probability
        if random.random() > 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

            # Flip bounding boxes
            img_width = img.shape[-1]
            new_boxes = []
            for bbox in labels['obj_bbox']:
                xmin, ymin, xmax, ymax = bbox

                # Flip xmin and xmax coordinates
                xmin_new = img_width - xmax
                xmax_new = img_width - xmin
                new_boxes.append([xmin_new, ymin, xmax_new, ymax])
                
            labels['obj_bbox'] = new_boxes

        return img, mask, labels
