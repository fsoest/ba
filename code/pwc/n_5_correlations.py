"""
Document to check for correlations between cell state and current rho
"""
import torch
import numpy as np
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding, angle_embedding
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
from torch.utils.data import DataLoader
from multiproc.pwc_helpers import wrapper, rho_path
import matplotlib.pyplot as plt
from rho_vis import exp_xyz, data_wrapper
from colour_bloch import Bloch as cBloch


N = 5
seed = 42
batch_size = 44
dt = 5
rho = 'eigen'
N_sobol = 45
runs = range(40)

# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

# %%
model = torch.load('models/N_5_rho_eigen_lstm').eval()

# %%
rhos = np.zeros((len(test_set), N, 2, 2), dtype=np.complex128)
for i, data in data_test:
    a, b, c = data_wrapper(data, dt, 1)
    rhos[i] = a
