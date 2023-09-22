import pandas as pd
import os

import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
    ):
        self.data_dir = data_dir
        self.train = train
        self.data = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.train:
            image = row.drop('label').values.reshape(28, 28)
            label = row['label']
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.int64)
            image = image / 255.0
            label = torch.nn.functional.one_hot(label, num_classes=10)
            return {
                'image': image.unsqueeze(0),  # (1, 28, 28)
                'label': label.float()  # (10)
            }

        else:
            image = row.values.reshape(28, 28)
            image = torch.tensor(image, dtype=torch.float32)
            image = image / 255.0
            return {
                'image': image.unsqueeze(0),  # (1, 28, 28)
            }
