"""
Generalisation to higher N of bi- and unidirectional N=5 networks
"""
import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
import numpy as np
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding
from rho_vis import rho_path
from qutip import rand_ket_haar as rkh
from multiproc.pwc_helpers import state_to_angles, get_eigen_rho
import matplotlib.pyplot as plt
from custom_loss import LSTMNetwork, work_loss
from lower_bound import lower_bound
# %%
N_max = 50
N_data = 1000
# %%
np.random.seed(42)
# Create random drives to N_max
N_arr = np.array(range(N_max)) + 2
rho_0 = np.zeros((N_data, 2, 2), dtype=np.complex128)
thetas = np.zeros((N_data, N_max + 2))
phis = np.zeros((N_data, N_max + 2))
for i in range(N_data):
    kets = np.zeros((N_max + 2, 2, 1), dtype=np.complex128)
    for j in range(N_max + 2):
        kets[j] = rkh(2).full()
    # Angles from state vectors
    thetas[i], phis[i] = state_to_angles(kets)
rho_0 = get_eigen_rho(thetas[:, 0], phis[:, 0])
# %%
# Make predictions and calculate work outputs
# uni = torch.load('models/dt_5_uni').eval()
uni = torch.load('models/N_5_rho_eigen_lstm_uni').eval()
# bi = torch.load('models/dt_5_bi').eval()
bi = torch.load('models/N_5_rho_eigen_lstm').eval()
uni_local = torch.load('models/local_opt/dt5_uni').eval()
# better = torch.load('models/custom_loss_dt_1_better').eval()
qcell = torch.load('models/q_cell_try').eval()
# %%
N = 5
seed = 42
batch_size = 44
dt = 5
rho = 'eigen'
N_sobol = 45
runs = range(5)


# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')
# %%

# Works: [N_data, N_max, N_max]
E_uni = np.zeros((N_data, N_max, N_max))
E_bi = np.zeros((N_data, N_max, N_max))
E_triv = np.zeros((N_data, N_max, N_max))
E_local = np.zeros((N_data, N_max, N_max))
E_better = np.zeros((N_data, N_max, N_max))
E_qcell = np.zeros((N_data, N_max, N_max))
# %%
for n in N_arr:
    x_embed = angle_embedding(np.concatenate((thetas[:, :n], phis[:, :n]), axis=1), n)
    x_embed = torch.from_numpy(x_embed)
    uni.N = n
    bi.N = n
    with torch.no_grad():
        hidden_uni, cell_uni = uni.HiddenCellTest(N_data)
        hidden_bi, cell_bi = bi.HiddenCellTest(N_data)
        hidden_local, cell_local = uni_local.HiddenCellTest(N_data)

        # h_better, c_better = qcell.HiddenCellTest(N_data)
        y_uni = uni(x_embed, hidden_uni, cell_uni)[0]
        y_bi = bi(x_embed, hidden_bi, cell_bi)[0]
        y_local = uni_local(x_embed, hidden_local, cell_local)[0]

        q_cell = qcell.init_cell(x_embed[:, 0])
        y_better = qcell(x_embed, q_cell)[0]

    # y_uni = rev_angle_embedding(y_uni.detach().numpy(), n)
    # y_bi = rev_angle_embedding(y_bi.detach().numpy(), n)
    # y_local = rev_angle_embedding(y_local.detach().numpy(), n)

    y_better = rev_angle_embedding(y_better.detach().numpy(), n)
    # Calculate energies
    for i in range(N_data):
        # E_uni[i, n-2, :n-1] = rho_path(thetas[i, :n], phis[i, :n], y_uni[i, :n], y_uni[i, n:], dt, rho_0[i], n, 1)[2][:-1]
        # E_bi[i, n-2, :n-1] = rho_path(thetas[i, :n], phis[i, :n], y_bi[i, :n], y_bi[i, n:], dt, rho_0[i], n, 1)[2][:-1]
        # E_local[i, n-2, :n-1] = rho_path(thetas[i, :n], phis[i, :n], y_local[i, :n], y_local[i, n:], dt, rho_0[i], n, 1)[2][:-1]
        # a = lower_bound(np.concatenate((thetas[i, :n], phis[i, :n])), n, dt)
        # E_triv[i, n-2, :n-1] = a[0][:-1]

        # E_better[i, n-2, :n-1] = rho_path(thetas[i, :n], phis[i, :n], y_better[i, :n], y_better[i, n:], dt, rho_0[i], n, 1)[2][:-1]
        E_qcell[i, n-2, :n-1] = rho_path(thetas[i, :n], phis[i, :n], y_better[i, :n], y_better[i, n:], dt, rho_0[i], n, 1)[2][:-1]



# np.save('gen/E_bi_dt_{0}'.format(dt), E_bi)
# np.save('gen/E_uni_dt_{0}'.format(dt), E_uni)
# np.save('gen/E_triv_dt_{0}'.format(dt), E_triv)
# np.save('gen/E_local_dt_{0}'.format(dt), E_local)
# np.save('gen/E_better_dt_1', E_better)
np.save('gen/E_qcell_dt_5', E_qcell)
# N = [2, 3, 4, 5, 9, 10]
# E_n = []
# # %%
dt = 5
E_bi = np.load('gen/E_bi_dt_{0}.npy'.format(dt))
E_uni = np.load('gen/E_uni_dt_{0}.npy'.format(dt))
E_triv = np.load('gen/E_triv_dt_{0}.npy'.format(dt))
E_local = np.load('gen/E_local_dt_{0}.npy'.format(dt))

# %%
# for n in N:
#     e = np.load('multi_train_data/dt/vardt_N_{0}_rho_eigen/dt_0_5_E_sobol_10_run_0.npy'.format(n))
#     E_n.append(np.mean(e[-1]))



# %%
# Visualisation
E_uni_sum = np.mean(np.cumsum(E_uni, axis=2)[:, :, -1], axis=0)
E_bi_sum = np.mean(np.cumsum(E_bi, axis=2)[:, :, -1], axis=0)
E_triv_sum = np.mean(np.cumsum(E_triv, axis=2)[:, :, -1], axis=0)
E_local_sum = np.mean(np.cumsum(E_local, axis=2)[:, :, -1], axis=0)
# E_better_sum = np.mean(np.cumsum(E_better, axis=2)[:, :, -1], axis=0)
plt.scatter(N_arr, -1 * E_uni_sum, label='Unidir. LSTM', marker='.')
plt.scatter(N_arr, -1 * E_bi_sum, label='Bidir. LSTM', marker='.')
plt.scatter(N_arr, -1 * E_triv_sum, label='Local opt.', marker='.')
plt.scatter(N_arr, -1 * E_local_sum, label='Local LSTM', marker='.')
# plt.scatter(N_arr, -1 * E_better_sum, label='Work loss', marker='.')
# plt.scatter(n_dt_1, -1 * np.array(E_n), label='Global opt.', marker='1', c='k')
plt.scatter(N_arr, -1 * E_q_cell_sum)

# plt.scatter(5, -1 * np.mean(data[:, 2]), marker='1')
plt.legend()
plt.xlabel('$N$')
plt.ylabel('$\overline{W}$')
# plt.savefig('/home/fsoest/ba/phystex/img/gen_dt_{0}.png'.format(dt), dpi=300)

# %%
# n_dt_1 = np.array([2, 3, 5, 12])
# E_n = np.zeros(4)
# E_n[-1] = np.mean(data[:, 2])
# E_n[0] *=-1
E_q_cell_sum = np.mean(np.cumsum(E_qcell, axis=2)[:, :, -1], axis=0)
plt.scatter(N_arr, -1 * E_q_cell_sum)
