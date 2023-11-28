import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageDataset
from models.u_net import UNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='UNetTraining',
        description="Train a UNet"
    )
    parser.add_argument('train_path')
    parser.add_argument('val_path', nargs='?', default='')
    parser.add_argument('save_path', nargs='?', default='model.pt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    train_ds = ImageDataset(args.train_path)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ds = ImageDataset(args.val_path)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    num_classes = 49

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_class=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1.0

    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            images = batch['image'].float().to(device)
            masks = batch['mask'].long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            loss = 0.0
            for batch in tqdm(val_loader):
                images = batch['image'].float().to(device)
                masks = batch['mask'].long().to(device)

                outputs = model(images)
                loss += criterion(outputs, masks).item() * images.size(0)
        loss /= len(val_ds)

        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Val Loss: {loss}")

        if loss < best_loss:
            best_loss = loss
            torch.save(model, args.save_path)

    print(f'Best Val Loss={best_loss}')
