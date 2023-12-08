import argparse
import os

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='OneHot',
        description="Do one-hot encoding on the given data"
    )
    parser.add_argument('data_path')
    parser.add_argument('--num_classes', type=int, default=49)
    args = parser.parse_args()

    video_dirs = [
        os.path.join(args.data_path, folder)
        for folder in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, folder)) and folder.startswith('video_')
    ]

    eye = np.eye(args.num_classes)
    for video_dir in tqdm(video_dirs):
        mask = np.load(os.path.join(video_dir, 'mask.npy'))
        mask = eye[mask]
        np.save(os.path.join(video_dir, 'one_hot.npy'), mask)
