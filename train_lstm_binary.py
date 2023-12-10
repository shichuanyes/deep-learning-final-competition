import argparse
import random

import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.video_dataset import VideoDataset
from models.lstm import Seq2Seq


def to_binary(masks: torch.Tensor) -> torch.Tensor:
    results = []
    for mask in masks:
        objects = torch.unique(mask[mask != 0])
        minibatch = mask.unsqueeze(0).repeat(len(objects), 1, 1, 1)
        for i, obj in enumerate(objects):
            minibatch[i][mask != obj] = 0
            minibatch[i][mask == obj] = 1
        results.append(minibatch)
    return torch.cat(results, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LSTMTraining',
        description="Train a LSTM (binary method)"
    )
    parser.add_argument('train_path')
    parser.add_argument('val_path', nargs='?', default='')
    parser.add_argument('save_path', nargs='?', default='model_lstm_binary.pt')
    parser.add_argument('--num_frames', type=int, default=11)
    parser.add_argument('--num_kernels', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    train_ds = VideoDataset(args.train_path)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ds = VideoDataset(args.val_path) if len(args.val_path) > 0 else train_ds
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    num_classes = 49

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seq2Seq(
        num_channels=1,
        num_kernels=args.num_kernels,
        kernel_size=3,
        padding=1,
        activation='relu',
        frame_size=(160, 240),
        num_layers=args.num_layers
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    scaler = torch.cuda.amp.GradScaler()

    best_score = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc='Train', leave=False):
            masks = to_binary(batch['masks'])

            idx = random.randrange(args.num_frames, train_ds.num_frames)
            inputs = masks[:, idx - args.num_frames:idx].float().to(device)
            target = masks[:, idx].float().to(device)

            inputs = inputs.unsqueeze(1)
            target = target.unsqueeze(1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(inputs)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        model.eval()
        with torch.inference_mode():
            score = 0.0
            for batch in tqdm(val_loader, desc='Validation', leave=False):
                masks = to_binary(batch['masks'])

                idx = random.randrange(args.num_frames, train_ds.num_frames)
                inputs = masks[:, idx - args.num_frames:idx, :, :].float().to(device)
                target = masks[:, idx, :, :].to(device)

                with torch.cuda.amp.autocast():
                    output = model(inputs)

                pred = torch.argmax(output, dim=1)
                score += jaccard(pred, target)

        print(f"Epoch [{epoch + 1}/{args.num_epochs}]: Train Loss: {train_loss / len(train_loader)} Val Score: {score / len(val_loader)}")

        if score > best_score:
            best_score = score
            torch.save(model, args.save_path)

    print(f'Best Val Score={best_score / len(val_loader)}')
