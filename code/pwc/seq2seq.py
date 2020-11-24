import torch
import torch.nn as nn
import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, N, d_model, wavelength = 10):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(N, d_model, requires_grad=False)
        for pos in range(N):
            for i in range(int(d_model / 2)):
                pe[pos, 2 * i] = np.sin(pos / wavelength**(2 * i / d_model))
                pe[pos, 2 * i + 1] = np.cos(pos / wavelength**(2 * i / d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x *= np.sqrt(self.d_model)
        x += self.pe
        return x
