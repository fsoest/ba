import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
import numpy as np
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding
# %%
N = 3
seed = 42
batch_size = 44
dt = 1
rho = 'eigen'
N_sobol = 15
runs = range(1)
# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
test_set = WorkDataset(data_test, N, 'lstm')
# %%
bi = torch.load('models/dt_1_bi').eval()
bi.N = N
bi.work_ratio(data_test, dt)
# %%
uni = torch.load('models/dt_1_uni').eval()
uni.N = N
uni.work_ratio(data_test, dt)
