from models import single_layer_fcANN, two_layer_fcANN
from dataset import WorkDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from multiproc.data_preprocessing import import_datasets, angle_embedding, rev_angle_embedding
import torch
import numpy as np
from multiproc.pwc_helpers import wrapper, rho_to_angles
# %%
N = 5
batch_size = 30
seed = 42
dt = 5
rho = 'eigen'
N_sobol = 45
runs = range(30)
neurons = [2000, 2000, 2000]
# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='ann')
test_set = WorkDataset(data_test, N, net='ann')
valid_set = WorkDataset(data_valid, N, net='ann')
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
# %%
# Only for rho haar with initial rhos
# for val in data:
#     theta, phi = rho_to_angles(val[3])
#     val[0][N-1] = theta
#     val[0][-1] = phi

# %%
torch.manual_seed(seed)
# model = single_layer_fcANN(10, N).double()
model = two_layer_fcANN(neurons, N, batch_size).double()
learning_rate = 1e-2
decay = 0.995
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimiser, decay)
# %%
model.learn(train_set, valid_set, optimiser, lr_schedule)
# %%
model = torch.load('best_model')

model.work_ratio(data_test, dt)
# %%
# Save model
torch.save(model.state_dict(), 'models/N_{}_rho_{}'.format(N, rho))

sum(p.numel() for p in model.parameters() if p.requires_grad)
