import os

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, num_frames: int = 22):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.num_videos = sum(
            os.path.isdir(os.path.join(self.data_dir, folder)) and folder.startswith('video_')
            for folder in os.listdir(self.data_dir)
        )
        folder = os.listdir(self.data_dir)[0]

    def __len__(self):
        return self.num_videos * self.num_frames

    def __getitem__(self, idx):
        img_idx = idx % self.num_frames
        video_dir = os.path.join(self.data_dir, f'video_{idx // self.num_frames}')
        img_dir = os.path.join(video_dir, f'image_{img_idx}.png')
        masks = np.load(os.path.join(video_dir, 'mask.npy'))
        return {
            'image': torchvision.io.read_image(img_dir),
            'mask': torch.tensor(masks[img_idx])
        }
