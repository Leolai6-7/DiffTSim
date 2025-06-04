import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, label_path=None, normalize=True):
        self.data = np.load(data_path)  # shape: (N, C, T)
        if label_path:
            self.labels = np.load(label_path)
        else:
            self.labels = np.zeros(len(self.data))
        if normalize:
            self.data = (self.data - self.data.mean(axis=(1,2), keepdims=True)) / (self.data.std(axis=(1,2), keepdims=True) + 1e-6)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
