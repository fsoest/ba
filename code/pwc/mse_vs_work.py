import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
import numpy as np
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding
from rho_vis import rho_path
import matplotlib.pyplot as plt
from colour_bloch import Bloch as cBloch

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
mse_test = WorkDataset(data_test, N, net='lstm')
work_test = WorkDataset(data_test[:, 0], N, net='custom_loss')


# %%
mse_model = torch.load('models/dt_1_bi')
from custom_loss import LSTMNetwork, work_loss
work_model = torch.load('models/custom_loss_dt_1_bi')

dict = mse_model.state_dict()
work_model.load_state_dict(dict)

work_model.work_ratio(data_test, dt)
work_model.work_ratio(data_valid, dt)
train_set = WorkDataset(data_train[:, 0], N, net='custom_loss')
valid_set = WorkDataset(data_valid[:, 0], N, net='custom_loss')

optimiser = torch.optim.SGD(work_model.parameters(), lr=0.02847, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.234, patience=55/3.7)

work_model.learn(train_set, valid_set, optimiser, scheduler)

# %%

work_model.work_ratio(data_test, dt)

from dataset import WorkDataset
work_model.calc_loss(work_test)
# %%
curr_arg = 80
inp = np.copy(data_test[curr_arg, 0])
emb_inp = torch.from_numpy(angle_embedding(inp[np.newaxis], N))
hidden, cell = mse_model.HiddenCellTest(1)
emb_out = mse_model(emb_inp, hidden, cell)[0].detach().numpy()
out = rev_angle_embedding(emb_out, 5)[0]
rhos_0, rho_0, E_0 = rho_path(inp[:N], inp[N:], out[:N], out[N:], dt, data_test[curr_arg, 3], N, 1)

inp_1 = np.copy(inp)
emb_inp_1 = torch.from_numpy(angle_embedding(inp_1[np.newaxis], N))
hidden, cell = work_model.HiddenCellTest(1)
emb_out_1 = work_model(emb_inp_1, hidden, cell)[0].detach().numpy()
out_1 = rev_angle_embedding(emb_out_1, 5)[0]
rhos_1, rho_1, E_1 = rho_path(inp_1[:N], inp_1[N:], out_1[:N], out_1[N:], dt, data_test[curr_arg, 3], N, 1)

rho_0_vec = np.zeros((N, 3))
h_sys_0 = np.zeros((N, 3))

rho_1_vec = np.zeros((N, 3))
h_sys_1 = np.zeros((N, 3))

h_drive_0 = np.zeros((N, 3))
h_drive_1 = np.zeros((N, 3))
for i in range(N):
    rho_0_vec[i] = np.array([2 * np.real(rhos_0[i][0, 1]), 2 * np.imag(rhos_0[i][1, 0]), rhos_0[i][0, 0] - rhos_0[i][1, 1]])
    h_sys_0[i] = np.sin(out[i]) *  np.array([np.cos(out[i + N]), np.sin(out[i + N]), 0])

    rho_1_vec[i] = np.array([2 * np.real(rhos_1[i][0, 1]), 2 * np.imag(rhos_1[i][1, 0]), rhos_1[i][0, 0] - rhos_1[i][1, 1]])
    h_sys_1[i] = np.sin(out_1[i]) *  np.array([np.cos(out_1[i + N]), np.sin(out_1[i + N]), 0])

    h_drive_0[i] = np.sin(inp[i]) *  np.array([np.cos(inp[i + N]), np.sin(inp[i + N]), 0])
    h_drive_1[i] = np.sin(inp_1[i]) *  np.array([np.cos(inp_1[i + N]), np.sin(inp_1[i + N]), 0])

fig = plt.figure(figsize=(15, 8))
for j in range(N-1):
    ax = fig.add_subplot(2, 4, j + 1, projection='3d')
    b = cBloch(fig=fig, axes=ax)
    b.add_points(rho_0_vec[j], colors='b')
    b.add_points(rho_0_vec[j+1], colors='k')
    b.add_vectors(h_drive_0[j], colors='r')
    b.add_vectors(h_sys_0[j], colors='y')
    b.add_vectors(h_sys_0[j+1], colors='g')
    b.render(fig=fig, axes=ax)

    ax = fig.add_subplot(2, 4, j + 5, projection='3d')
    b = cBloch(fig=fig, axes=ax)
    b.add_points(rho_1_vec[j], colors='b')
    b.add_points(rho_1_vec[j+1], colors='k')
    b.add_vectors(h_drive_1[j], colors='r')
    b.add_vectors(h_sys_1[j], colors='y')
    b.add_vectors(h_sys_1[j+1], colors='g')
    b.render(fig=fig, axes=ax)
np.cumsum(E_0)[-1]
np.cumsum(E_1)[-1]
