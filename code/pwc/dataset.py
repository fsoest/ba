import numpy as np
from torch.utils.data import Dataset
import torch
from multiproc.data_preprocessing import angle_embedding


class WorkDataset(Dataset):
    """ Custom Dataset to import """

    def __init__(self, data, N, reshape=False, embed=True):
        """
        data: numpy array
        """
        self.embed = embed
        self.data = data
        self.N = N
        self.reshape = reshape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.embed:
            try:
                iter(idx)
                x = torch.from_numpy(angle_embedding(self.data[idx, 0], self.N, reshape=self.reshape))
                y = torch.from_numpy(angle_embedding(self.data[idx, 1], self.N, reshape=self.reshape))
                # Full ones to indicate end of sequence
                if self.reshape == False:
                    x[:, -1] = torch.full((len(idx), 4), 1, dtype=torch.float64)
            except:
                x = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 0], self.N, reshape=self.reshape))
                y = torch.from_numpy(angle_embedding(self.data[np.newaxis, idx, 1], self.N, reshape=self.reshape))
                # Full ones to indicate end of sequence
                if self.reshape == False:
                    x[:, -1] = torch.full((1, 4), 1, dtype=torch.float64)
        else:
            x = self.data[idx, 0]
            y = self.data[idx, 1]

        sample = {'x': x, 'y': y}

        return sample
