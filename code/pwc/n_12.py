import torch
import numpy as np
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
from torch.utils.data import DataLoader


N = 12
seed = 42
batch_size = 44
dt = 5
rho = 'eigen'
N_sobol = 10
runs = range(232)

# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='ann')
test_set = WorkDataset(data_test, N, net='ann')
valid_set = WorkDataset(data_valid, N, net='ann')
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

len(data_train)/len(data)

len(data_test)/len(data)

len(data_valid)/len(data)

train_set[0]['']


# %%
model = torch.load('models/N_5_rho_eigen_lstm')
model.N = 12

mean1 = model.work_ratio(data_test[:10000], dt)
mean2 = model.work_ratio(data_test[10000:20000], dt)
mean3 = model.work_ratio(data_test[20000:], dt)

(mean1 + mean2 + mean3)/3

# %%
model2 = torch.load('models/N_5_rho_eigen_lstm_uni')
model.N = 12

mean1 = model.work_ratio(data_test[:10000], dt)
mean2 = model.work_ratio(data_test[10000:20000], dt)
mean3 = model.work_ratio(data_test[20000:], dt)
