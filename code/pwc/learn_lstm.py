import numpy as np
from models import LSTMNetwork
from dataset import WorkDataset
import torch
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding
from sklearn.model_selection import train_test_split
from multiproc.pwc_helpers import wrapper



batch_size = 30
seed = 42
learning_rate = 1e-2
n_layers = 2
# %%
N = 12
dt = 5
N_sobol = 10
runs = [0, 1]#, 2, 3, 4]
rho = 'eigen'

data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, embed=True)
test_set = WorkDataset(data_test, N, embed=True)
valid_set = WorkDataset(data_valid, N, embed=True)

torch.manual_seed(seed)
model = LSTMNetwork(4, 4, batch_size, n_layers, N)
model = model.double()


optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

# %%
torch.device('cpu')
model.train(train_set, valid_set, optimiser, patience=30)


# %%
100*model.work_ratio(data_test, 5)
