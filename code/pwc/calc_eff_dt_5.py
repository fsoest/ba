import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
from train_lstm import train_lstm_total_dropout as train_lstm
import numpy as np
from lower_bound import lower_bound

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

# %%
ann = torch.load('models/N_5_ann')
ann
ann.work_ratio(data_test, dt)
ann.calc_loss(test_set)
dt
# %%
bi = torch.load('models/N_5_rho_eigen_lstm').eval()
with torch.no_grad():
    biwr = bi.work_ratio(data_test, dt)
    biloss = bi.calc_loss(test_set)

# %%
uni = torch.load('models/N_5_rho_eigen_lstm_uni').eval()
with torch.no_grad():
    uniwr = uni.work_ratio(data_test, dt)
    uniloss = uni.calc_loss(test_set)

# %%
e_min = np.zeros(len(data))
for i, d in enumerate(data):
    a = lower_bound(d[0], N, dt)
    e_min[i] = np.cumsum(a[0])[-1]

np.mean(e_min)
np.mean(data_test[:, 2])
