import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.video_dataset import VideoDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PredictMask',
        description="Predict image masks"
    )
    parser.add_argument('data_path')
    parser.add_argument('model_path', nargs='?', default='model.pt')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    model = torch.load(args.model_path)

    ds = VideoDataset(args.data_path)
    loader = DataLoader(ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader):
            images = batch['images'].float().to(device)
            paths = batch['path']

            output = model(images.view(-1, images.size(2), images.size(3), images.size(4)))
            output = torch.argmax(output, dim=1)

            for mask, path in zip(output.view(images.size(0), images.size(1), images.size(3), images.size(4)), paths):
                np.save(os.path.join(path, 'mask.npy'), mask.cpu().numpy())
