import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
from train_lstm import train_lstm_total_dropout as train_lstm
from custom_loss import LSTMNetwork, work_loss


N = 5
seed = 42
batch_size = 44
dt = 1
rho = 'eigen'
N_sobol = 45
runs = range(21)
net = 'custom-loss'
# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net=net)
test_set = WorkDataset(data_test, N, net=net)
valid_set = WorkDataset(data_valid, N, net=net)
# %%
dt_1_bi = torch.load('models/custom_loss_dt_1_bi')
dt_1_bi.work_ratio(data_test, dt)

dt_1_uni = torch.load('models/custom_loss_dt_1_uni')
dt_1_uni.work_ratio(data_test, dt)

# %%
N = 5
seed = 42
batch_size = 44
dt = 5
rho = 'eigen'
N_sobol = 45
runs = range(40)
net = 'custom-loss'
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net=net)
test_set = WorkDataset(data_test, N, net=net)
valid_set = WorkDataset(data_valid, N, net=net)
# %%
dt_5_bi = torch.load('models/custom_loss_dt_5_bi')
dt_5_bi.work_ratio(data_test, dt)
# %%
dt_5_uni = torch.load('models/custom_loss_dt_5_uni')
dt_5_uni.work_ratio(data_test, dt)
dt
