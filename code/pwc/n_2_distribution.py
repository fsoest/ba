import torch
import numpy as np
import matplotlib.pyplot as plt
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
from models import single_layer_fcANN
from multiproc.pwc_helpers import wrapper, rho_to_angles, rho_path
from rho_vis import data_wrapper


N = 5
seed = 42
dt = 5
rho = 'eigen'
N_sobol = 45
runs = range(40)

data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)

# Only for rho haar with initial rhos
# for val in data:
#     theta, phi = rho_to_angles(val[3])
#     val[0][N-1] = theta
#     val[0][-1] = phi

data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')

# %%
import torch

from custom_loss import work_loss

loss = work_loss(5, 3)
d = train_set.__getitem__(0)
# %%

t = np.load('multiproc/train_data/N_3/dt_5_eigen_sobol_1_run_0.npy', allow_pickle=True)


from multiproc.data_preprocessing import angle_embedding
t[0][0]
x = angle_embedding(t[0, 0, np.newaxis], 3)
y = angle_embedding(t[0, 1, np.newaxis], 3)
t[0, 2]
# %%
a = loss.forward(torch.from_numpy(y), torch.from_numpy(x))


#%%
a = torch.full((1,), 0.7628+0.5j, dtype=torch.cfloat)
b = torch.full((1, ), 0.6466)
torch.exp(a)
torch.atan2(c[:, :, 1], c[:, :, 3]) % (2*np.pi)
c = train_set.__getitem__(range(2))['x']
c
a = np.zeros((3, 2, 2), dtype=np.complex128)
for i in range(3):
    a[i] = data_train[i, 3]
from torch._vmap_internals import vmap


tens = torch.from_numpy(a)
tens
test = torch.from_numpy(np.array([[[1+2j, 3+4j], [5-6j, 7-8j]], [[5-34j, -8j], [10+1j, 6-4j]]]))

b = torch.transpose(test, 1, 2).conj()
b
map_trace = vmap(torch.trace)

857/21


mse = torch.nn.MSELoss()
# %%
%%timeit
mse(d['y'], d['x'])
# %%
map_trace(b.real)

torch.trace(b.real[0])

b[:, 0, 0] + b[:, 1, 1]

torch.trace(torch.real(b))

torch.trace(tens)

test[1].T.conj()

tens[0] @ tens[0]
torch.matmul(tens, tens)
torch.trace(tens[])
tens[:, 0, 0] + tens[:, 1, 1]
map = vmap(torch.matrix_exp)
map(tens)
tens @ tens

tens[2] @ tens[2]
torch.matrix_exp(tens[0])

train_set.__getitem__(range(2))['x'][:, 0, 1]

# %%
# model = single_layer_fcANN(10, N).double()
# model.load_state_dict(torch.load('models/N_{}_rho_{}'.format(N, rho)))
model = torch.load('models/N_5_ann').eval()
hidden, cell = model.HiddenCellTest(14400)#len(test_set))
# Make predictions
with torch.no_grad():
    y_pred = model(test_set.__getitem__(range(14400))['x'])#, hidden, cell)
# Reverse embedding
trans_pred = rev_angle_embedding(y_pred.detach().numpy(), N, reshape=True)
# Calculate work output
pred_work = np.zeros(len(trans_pred))
for i, pred in enumerate(trans_pred):
    pred_work[i] = wrapper(pred, data_test[i, 0][:N], data_test[i, 0][N:], dt, data_test[i, 3], N)

# %%
import pandas as pd
import seaborn as sns
lstm_bi = pd.DataFrame()
lstm_bi['W'] = -1 * data_test[:, 2]
lstm_bi['$\eta$'] = pred_work / data_test[:, 2]
# %%
np.mean(pred_work/data_test[:, 2])
sns.jointplot(lstm_bi['W'], lstm_bi['$\eta$'], alpha=0.1)
plt.ylim(-1.1, 1.1)
plt.savefig('/home/fsoest/ba/phystex/img/work_dist_n5_{}_ann.png'.format(rho), dpi=300)
