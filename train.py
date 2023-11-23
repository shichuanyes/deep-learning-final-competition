import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageDataset
from model import UNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='UNetTraining',
        description="Train a UNet"
    )
    parser.add_argument('train_path')
    parser.add_argument('save_path', nargs='?', default='model.pt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    loader = DataLoader(ImageDataset(args.train_path), batch_size=args.batch_size, shuffle=False)

    num_classes = 49
    train_loader = loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_class=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        for batch in tqdm(train_loader):
            images = batch['image'].float().to(device)
            masks = batch['mask'].long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss.item()}")

    torch.save(model, args.save_path)
