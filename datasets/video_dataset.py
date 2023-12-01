import os

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_dir: str, num_frames: int = 22):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.video_dirs = [
            os.path.join(self.data_dir, folder)
            for folder in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, folder)) and folder.startswith('video_')
        ]

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        images = [
            torchvision.io.read_image(os.path.join(video_dir, f'image_{img_idx}.png'))
            for img_idx in range(self.num_frames)
        ]
        masks = np.load(os.path.join(video_dir, 'mask.npy'))
        return {
            'images': torch.stack(images),
            'masks': torch.tensor(masks).long()
        }