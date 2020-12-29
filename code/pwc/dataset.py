import numpy as np
from torch.utils.data import Dataset
import torch
from multiproc.data_preprocessing import angle_embedding, mult_embedding


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
                # x = np.zeros((len(idx), self.N, 2))
                # # Reshape y into (batch, N, 2)
                # y = np.zeros((len(idx), self.N, 2))
                # for i in range(len(idx)):
                #     x[i] = self.data[i, 0].reshape(2, self.N).T
                 #     y[i] = self.data[i, 1].reshape(2, self.N).T
                # x = torch.from_numpy(x)
                # y = torch.from_numpy(y)
            except:
                x = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 0], self.N))
                y = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 1], self.N))
                # x = self.data[idx, 0].reshape((1, 2, self.N))
                # x = x.transpose(0, 2, 1)
                # x = torch.from_numpy(x)
                # y = self.data[idx, 1].reshape((1, 2, self.N))
                # y = y.transpose(0, 2, 1)
                # y = torch.from_numpy(y)
        elif self.net == 'ann':
            try:
                iter(idx)
                x = torch.from_numpy(angle_embedding(self.data[idx, 0], self.N, reshape=True))
                y = torch.from_numpy(angle_embedding(self.data[idx, 1], self.N, reshape=True))
                # x = torch.from_numpy(angle_embedding(self.data[idx, 0], self.N))
                # # Reshape y into (batch, N, 2)
                # y = np.zeros((len(idx), self.N, 2))
                # for i in range(len(idx)):
                #     y[i] = self.data[i, 1].reshape(2, self.N).T
                # y = torch.from_numpy(y)
            except:
                x = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 0], self.N, reshape=True))
                y = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 1], self.N, reshape=True))
                # x = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 0], self.N))
                # y = self.data[idx, 1].reshape((1, 2, self.N))
                # y = y.transpose(0, 2, 1)
                # y = torch.from_numpy(y)
        elif self.net == 'mult_lstm':
            try:
                iter(idx)
                x = torch.from_numpy(mult_embedding(self.data[idx, 0], self.N))
                y = torch.from_numpy(mult_embedding(self.data[idx, 1], self.N))
            except:
                x = torch.from_numpy(mult_embedding(self.data[np.newaxis, idx, 0], self.N))
                y = torch.from_numpy(mult_embedding(self.data[np.newaxis, idx, 1], self.N))
        sample = {'x': x, 'y': y}

        return sample
