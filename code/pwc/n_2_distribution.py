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
train_set = WorkDataset(data_train, N, net='custom_loss')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')
# %%

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
