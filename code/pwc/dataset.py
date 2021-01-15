import numpy as np
from torch.utils.data import Dataset
import torch
from multiproc.data_preprocessing import angle_embedding, mult_embedding, out_embedding


class WorkDataset(Dataset):
    def __init__(self, data, N, net):
        self.data = data
        self.N = N
        self.net = net


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.net == 'lstm':
            try:
                iter(idx)
                x = torch.from_numpy(angle_embedding(self.data[idx, 0], self.N))
                y = torch.from_numpy(angle_embedding(self.data[idx, 1], self.N))
            except:
                x = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 0], self.N))
                y = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 1], self.N))
        elif self.net == 'ann':
            try:
                iter(idx)
                x = torch.from_numpy(angle_embedding(self.data[idx, 0], self.N, reshape=True))
                y = torch.from_numpy(angle_embedding(self.data[idx, 1], self.N, reshape=True))
            except:
                x = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 0], self.N, reshape=True))
                y = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 1], self.N, reshape=True))
        elif self.net == 'custom_loss':
            try:
                iter(idx)
                x = torch.from_numpy(angle_embedding(self.data[idx], self.N))
            except:
                x = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx], self.N))

        try:
            sample = {'x': x, 'y': y}
        except:
            sample = {'x': x}

        return sample
