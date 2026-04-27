import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *

def transform(image):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = arr.transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name)
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open_rgb(image_path)
        return transform(image), torch.tensor(np.array(segment_image), dtype=torch.float32)
