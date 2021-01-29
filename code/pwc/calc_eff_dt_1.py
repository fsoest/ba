import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
from train_lstm import train_lstm_total_dropout as train_lstm
# from models import LSTM_total_dropout as LSTMNetwork
import numpy as np
from lower_bound import lower_bound

N = 5
seed = 42
batch_size = 44
dt = 1
rho = 'eigen'
N_sobol = 45
runs = range(21)

# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')

# %%
# Unidirectional Network
uni = torch.load('models/dt_1_uni').eval()
uni.work_ratio(data_test, dt)
sum(p.numel() for p in uni.parameters() if p.requires_grad)
uni.calc_loss(test_set)

bi = torch.load('models/dt_1_bi').eval()
bi.work_ratio(data_valid, dt)
bi.work_ratio(data_test, dt)
sum(p.numel() for p in bi.parameters() if p.requires_grad)
bi.calc_loss(test_set)

# %%
e_min = np.zeros(len(data_test))
for i, d in enumerate(data_test):
    a = lower_bound(d[0], N, dt)
    e_min[i] = np.cumsum(a[0])[-1]

np.mean(e_min)
np.mean(data_test[:, 2])
