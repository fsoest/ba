from models import single_layer_fcANN, two_layer_fcANN
from dataset import WorkDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from multiproc.data_preprocessing import import_datasets, angle_embedding, rev_angle_embedding
import torch
import numpy as np
from multiproc.pwc_helpers import wrapper, rho_to_angles
# %%
N = 12
batch_size = 44
seed = 42
dt = 5
rho = 'eigen'
N_sobol = 10
runs = range(31)
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

patience = 55
pat_drop = 3.698498258242224
sched_factor = 0.24509238889070978
dropout = 0.3179732914255167
# %%
torch.manual_seed(seed)
# model = single_layer_fcANN(10, N).double()
model = two_layer_fcANN(neurons, N, batch_size, dropout).double()
learning_rate = 0.02847560288866327

criterion = torch.nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)

model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)
model.eval().work_ratio(data_test, dt)

torch.save(model, 'models/N_5_ann')
len(data[27000, 0])
len(np.load('multi_train_data/N_12/dt_5_eigen_sobol_10_run_31.npy', allow_pickle=True))

# %%
# Save model

sum(p.numel() for p in model.parameters() if p.requires_grad)

model = torch.load('models/N_5_rho_eigen_lstm_uni')
model.eval().work_ratio(data_test, dt)
model
model.state_dict()

model

model = torch.load('models/N_5_rho_eigen_lstm')
model
