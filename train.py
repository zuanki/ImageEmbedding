
import os
import time
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.datasets.mnist_dataset import MNISTDataset
from src.models.DAE import DenoisingAutoencoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data', help='data directory')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints', help='save directory')

    args = parser.parse_args()

    return args


def main(args):
    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LR = args.lr
    SAVE_DIR = args.save_dir
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    dataset = MNISTDataset(data_dir=DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DenoisingAutoencoder(latent_dim=2).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    result = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        epoch_loss = 0.0
        for batch in dataloader:
            image = batch['image'].to(DEVICE)
            label = batch['label'].to(DEVICE)

            optimizer.zero_grad()

            pred = model(image)
            loss = criterion(pred, image)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        result.append({
            'epoch': epoch,
            'loss': epoch_loss / len(dataloader)
        })

    torch.save(model.state_dict(), os.path.join(
        SAVE_DIR, f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'))

    df_result = pd.DataFrame(result)

    # Save training results
    df_result.to_csv(os.path.join(
        SAVE_DIR, f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
