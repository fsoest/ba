import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
import numpy as np
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding
from rho_vis import rho_path
# %%
N = 12
seed = 42
batch_size = 44
dt = 1
rho = 'eigen'
N_sobol = 45
runs = range(5)
# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
test_set = WorkDataset(data_test, N, 'lstm')
# %%
bi = torch.load('models/dt_1_bi').eval()
bi.N = N
with torch.no_grad():
    x = bi.work_ratio(data_test, dt)

uni = torch.load('models/dt_1_uni').eval()
uni.N = N
with torch.no_grad():
    y = uni.work_ratio(data_test, dt)


x
y



with torch.no_grad():
    uni_hidden, uni_cell = uni.HiddenCellTest(len(data_test))
    d = test_set.__getitem__(range(len(data_test)))
    uni_pred = uni(d['x'], uni_hidden, uni_cell)[0]
    bi_hidden, bi_cell = bi.HiddenCellTest(len(data_test))
    bi_pred = bi(test_set.__getitem__(range(len(data_test)))['x'], bi_hidden, bi_cell)[0]

uni_trans = rev_angle_embedding(uni_pred, N)
bi_trans = rev_angle_embedding(bi_pred, N)

E_uni = np.zeros((len(data_test), 12))
E_bi = np.zeros((len(data_test), 12))
for i, data in enumerate(data_test):
    E_uni[i] = rho_path(data[0][:N], data[0][N:], uni_trans[i, :N], uni_trans[i, N:], dt, data[3], N, 1)[2]
    E_bi[i] = rho_path(data[0][:N], data[0][N:], bi_trans[i, :N], bi_trans[i, N:], dt, data[3], N, 1)[2]

E_bi.shape
import matplotlib.pyplot as plt
plt.plot(range(12), -1*E_bi.T)
# %%
E_bi.shape
np.mean(E_bi, axis=0).shape
plt.scatter(range(12), -1 * np.mean(E_bi, axis=0))
plt.hlines(0, 12, 0)
# %%
np.cumsum(E_bi, axis=1)[:, -1].shape
np.mean(np.cumsum(E_bi, axis=1)[:, -1])
# %%
E_bi.shape
np.mean(E_uni, axis=0).shape
plt.scatter(range(12), -1 * np.mean(E_uni, axis=0))
plt.hlines(0, 12, 0)
