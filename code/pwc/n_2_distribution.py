import torch
import numpy as np
import matplotlib.pyplot as plt
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
from models import single_layer_fcANN
from multiproc.pwc_helpers import wrapper, rho_to_angles


N = 2
seed = 42
dt = 5
rho = '0'
N_sobol = 10
runs = range(1)

data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)

# Only for rho haar with initial rhos
# for val in data:
#     theta, phi = rho_to_angles(val[3])
#     val[0][N-1] = theta
#     val[0][-1] = phi

data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='ann')
test_set = WorkDataset(data_test, N, net='ann')
valid_set = WorkDataset(data_valid, N, net='ann')

model = single_layer_fcANN(10, N).double()
model.load_state_dict(torch.load('models/N_{}_rho_{}'.format(N, rho)))
model.eval()

trans_pred = rev_angle_embedding(model(test_set.__getitem__(range(len(test_set)))['x']).detach().numpy(), N, reshape=True)

pred_work = np.zeros(len(trans_pred))
for i, pred in enumerate(trans_pred):
    pred_work[i] = wrapper(pred, data_test[i, 0][:N], data_test[i, 0][N:], dt, data_test[i, 3], N)

data_test[:, 1]



data = np.zeros((len(data_test), 2 * N))
for i, d in enumerate(data_test[:, 1]):
    data[i] = d
# %%
from multiproc.data_preprocessing import angles_to_cart
cart = angles_to_cart(data)
red = tsne.fit_transform(cart)

# %%
# plt.scatter(-1*data_test[:, 2], pred_work/data_test[:, 2])
# plt.ylabel('$\eta$')
# plt.xlabel('W')
#
# plt.hist(pred_work/data_test[:, 2], bins=100)
plt.scatter(red[:, 0], red[:, 1], c=data_test[:, 2])
plt.colorbar()

# %%
import pandas as pd

a = pd.DataFrame()
a['W'] = -1 * data_test[:, 2]
a['eta'] = pred_work/data_test[:, 2]
# sns.set_theme(style="darkgrid")

# %%
ax1 = plt.subplot(311)
ax1.scatter(a['W'], a['eta'])
ax1.set_xlabel('$W_{opt}$')
ax1.set_ylabel('$\eta$')

# %%

plt.scatter(a['W'], a['eta'])
plt.xlabel('$W_{opt}$')
plt.ylabel('$\eta$')
plt.savefig('/home/fsoest/ba/phystex/img/work_dist_n2_{}.png'.format(rho), dpi=300)

# %%
ax2 = plt.subplot(211)
ax2.hist(a['W'], bins=100)
ax2.set_xlabel('$W_{opt}$')

ax3 = plt.subplot(212)
# ax3.vlines(np.mean(a['eta']), 0, 220, colors='red')
ax3.hist(a['eta'], bins=100)
ax3.set_xlabel('$\eta$')

plt.tight_layout()
plt.savefig('/home/fsoest/ba/phystex/img/hist_n2_{}.png'.format(rho), dpi=300)
