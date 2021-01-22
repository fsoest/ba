import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
from train_lstm import train_lstm_total_dropout as train_lstm
# from models import LSTM_total_dropout as LSTMNetwork

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
train_set = WorkDataset(data_train, N, net='ann')
test_set = WorkDataset(data_test, N, net='ann')
valid_set = WorkDataset(data_valid, N, net='ann')

# %%
ann = torch.load('models/N_5_ann')
ann
ann.work_ratio(data_test, dt)
ann.calc_loss(test_set)
dt
# %%
bi = torch.load('models/N_5_rho_eigen_lstm')
bi.work_ratio(data_test, dt)
bi.calc_loss(test_set)

# %%
uni = torch.load('models/N_5_rho_eigen_lstm_uni')
uni.work_ratio(data_test, dt)
uni.calc_loss(test_set)
