# RUPERT MENNEER (2024)

from training.dataset import Dataset
import torchvision
from datasets import load_dataset

#----------------------------------------------------------------------------
# Dataset subclass that loads images + conditionals from the specified hugging face dataset

class HuggingFaceDataset(Dataset):
    def __init__(self,
        dataset_name,                   # Name of the dataset on hugging face e.g. limingcv/Captioned_COCOStuff.
        split='train',                  # Name of the split e.g. train or validation or train[:10]. Default is train.
        target_resolution=(256, 256),   # Target resolution (H, W) to resize and crop images.
        **super_kwargs,                 # Additional arguments for the Dataset base class.
    ):
        
        self.dataset = load_dataset(dataset_name, split=split)
        self._target_resolution = target_resolution
        self._raw_shape = [len(self.dataset), 3, *target_resolution]
        super().__init__(name=dataset_name, raw_shape=self._raw_shape, **super_kwargs)

    def get_standard_image_transform(self, interpolation=torchvision.transforms.InterpolationMode.BICUBIC):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(self._target_resolution, interpolation=interpolation),  # resize the shorter side to min target res, retaining aspect ratio
            torchvision.transforms.CenterCrop(self._target_resolution),  # center crop to target res
        ])

    def _load_raw_image(self, raw_idx): # to be overridden by subclass for specific dataset
        raise NotImplementedError

    def _preprocess(self): # to be overridden by subclass for specific dataset
        raise NotImplementedError
    
    def _load_raw_labels(self): # to be overridden by subclass for specific dataset
        raise NotImplementedError

    def __getitem__(self, idx): # to be overridden by subclass for specific dataset
        raise NotImplementedError


