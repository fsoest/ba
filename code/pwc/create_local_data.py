import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from lower_bound import lower_bound
import numpy as np
# %%
N = 5
seed = 42
batch_size = 44
dt = 5
rho = 'eigen'
N_sobol = 45
runs = range(40)
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)

# %%
dt = 0.2
local_data = np.copy(data)
for d in local_data:
    E_arr, theta, phi = lower_bound(d[0], N, dt)
    d[1] = np.concatenate((theta, phi))
    d[2] = np.cumsum(E_arr)[-1]

np.mean(local_data[:, 2])



np.save('local_opt_data/N_{0}_dt_{1}'.format(N, dt), local_data)
