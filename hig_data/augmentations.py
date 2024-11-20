import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch

class HIGAugmentation(object):
    def __init__(self, size=512, train=True):
        self.size = size
        self.train = train

        # Spatial augmentations that affect image, mask, and bounding boxes
        self.random_crop = self.random_resized_crop
        self.hflip = self.random_horizontal_flip
        # Color augmentations that only affect the image
        self.augment = T.Compose([T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)])

        self.center_crop_resize = self.center_crop

    def __call__(self, img, mask, labels):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if self.train:
            # Apply spatial augmentations to img, mask, and bounding boxes
            img, mask, labels = self.random_crop(img, mask, labels)
            img, mask, labels = self.hflip(img, mask, labels)
            # Apply color augmentations to the image only
            img = self.augment(img)
        else:
            # if val sample then just center crop
            img, mask, labels = self.center_crop_resize(img, mask, labels)
        
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
        for key in ['obj_bbox', 'obj_class']:
            labels[key] = [labels[key][idx] for idx in valid_idxs]
        labels['obj_bbox'] = new_boxes  # Update with new bounding boxes

        return img, mask, labels

    def center_crop(self, img, mask, labels):
        # Calculate center crop parameters
        width, height = img.shape[-1], img.shape[-2]
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2

        # Perform the center crop
        img = F.crop(img, top, left, crop_size, crop_size)
        mask = F.crop(mask, top, left, crop_size, crop_size)
        # Resize to the target size
        img = F.resize(img, (self.size, self.size), interpolation=T.InterpolationMode.BICUBIC)
        mask = F.resize(mask, (self.size, self.size), interpolation=T.InterpolationMode.NEAREST)

        # Threshold for minimum bounding box area (2% of cropped image area)
        min_area = 0.02 * self.size * self.size
        new_boxes = []
        valid_idxs = []

        # Adjust bounding boxes based on crop
        for idx, bbox in enumerate(labels['obj_bbox']):
            xmin, ymin, xmax, ymax = bbox

            # Adjust box coordinates based on crop
            xmin_new = ((xmin - left) / crop_size) * self.size
            ymin_new = ((ymin - top) / crop_size) * self.size
            xmax_new = ((xmax - left) / crop_size) * self.size
            ymax_new = ((ymax - top) / crop_size) * self.size

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
        for key in ['obj_bbox', 'obj_class']:
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



class BatchedHIGAugmentation(object):
    def __init__(self, size=512, train=True):
        self.size = size
        self.train = train
        self.augment = T.Compose([T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.03)])

    def __call__(self, batch):

        # imgs, masks, labels = batch
        
        imgs = torch.stack([torch.from_numpy(i[0]) for i in batch]) # Assumes center cropped and same res images
        masks = torch.stack([torch.from_numpy(i[1]) for i in batch])
        labels = [i[2] for i in batch]

        # Convert numpy arrays to torch tensors if needed
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)

        if self.train:
            # Apply spatial augmentations to all images and masks in the batch
            imgs, masks, labels = self.random_resized_crop(imgs, masks, labels)
            imgs, masks, labels = self.random_horizontal_flip(imgs, masks, labels)
            imgs = self.augment(imgs)  # Apply color augmentations to batch of images            
        
        return imgs, masks, labels

    # applies one random resized crop across entire image batch (massive speed-up)
    def random_resized_crop(self, imgs, masks, labels):
        
        # Generate parameters for cropping (one set per batch)
        i, j, h, w = T.RandomResizedCrop.get_params(imgs[0], scale=(0.65, 1), ratio=(1., 1.))

        imgs = F.resized_crop(imgs, i, j, h, w, (self.size, self.size), interpolation=T.InterpolationMode.BICUBIC)
        masks = F.resized_crop(masks, i, j, h, w, (self.size, self.size), interpolation=T.InterpolationMode.NEAREST)
        labels = self._crop_box(labels, (i, j, h, w))

        return imgs, masks, labels

    def _crop_box(self, all_labels, crop_params):

        i, j, h, w = [i/self.size for i in crop_params]


        # Threshold for minimum bounding box area (2% of cropped image area)
        min_area = 0.02 

        out = []

        # Adjust bounding boxes based on crop
        for labels in all_labels:
            new_boxes = []
            valid_idxs = []
            for idx, bbox in enumerate(labels['obj_bbox']):
                xmin, ymin, xmax, ymax = bbox

                # Adjust box coordinates based on crop
                xmin_new = ((xmin - j) / w)
                ymin_new = ((ymin - i) / h) 
                xmax_new = ((xmax - j) / w) 
                ymax_new = ((ymax - i) / h)

                # Clamp to ensure coordinates are within [0, self.size]
                xmin_new = max(0, min(1, xmin_new))
                ymin_new = max(0, min(1, ymin_new))
                xmax_new = max(0, min(1, xmax_new))
                ymax_new = max(0, min(1, ymax_new))
                
                # Calculate new width and height after adjustment
                bw_new = xmax_new - xmin_new
                bh_new = ymax_new - ymin_new

                # Check if the new box area meets the minimum threshold
                if bw_new * bh_new >= min_area:
                    new_boxes.append([xmin_new, ymin_new, xmax_new, ymax_new])
                    valid_idxs.append(idx)  # Store the original index of valid boxes

            # Filter labels using valid_idxs
            for key in ['obj_bbox', 'obj_class']:
                labels[key] = [labels[key][idx] for idx in valid_idxs]
            labels['obj_bbox'] = new_boxes  # Update with new bounding boxes
            out.append(labels)

        return out

    def random_horizontal_flip(self, imgs, masks, labels):
        if random.random() > 0.5:
            imgs = F.hflip(imgs)
            masks = F.hflip(masks)
            labels = self._horizontal_flip_box(labels)

        return imgs, masks, labels

    def _horizontal_flip_box(self, all_labels):
        out = []
        for labels in all_labels:
            img_width = 1
            new_boxes = []
            for bbox in labels['obj_bbox']:
                xmin, ymin, xmax, ymax = bbox

                # Flip xmin and xmax coordinates
                xmin_new = img_width - xmax
                xmax_new = img_width - xmin
                new_boxes.append([xmin_new, ymin, xmax_new, ymax])
            labels['obj_bbox'] = new_boxes
            out.append(labels)
        return out


    def center_crop(self, labels):
        # Calculate center crop parameters
        width, height = self.size, self.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2

        labels = self._crop_box(labels, (top, left, height, width))

        return labels


    