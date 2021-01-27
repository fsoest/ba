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
# %%
N_max = 50
N_data = 1000
dt = 1
# %%
# Create random drives to N_max
N = np.array(range(N_max)) + 2
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
uni = torch.load('models/dt_1_uni').eval()
bi = torch.load('models/dt_1_bi').eval()

# Works: [N_data, N_max, N_max]
E_uni = np.zeros((N_data, N_max, N_max))
E_bi = np.zeros((N_data, N_max, N_max))

# %%
for n in N:
    x_embed = angle_embedding(np.concatenate((thetas[:, :n], phis[:, :n]), axis=1), n)
    x_embed = torch.from_numpy(x_embed)
    uni.N = n
    bi.N = n
    with torch.no_grad():
        hidden_uni, cell_uni = uni.HiddenCellTest(N_data)
        hidden_bi, cell_bi = bi.HiddenCellTest(N_data)
        y_uni = uni(x_embed, hidden_uni, cell_uni)[0]
        y_bi = bi(x_embed, hidden_bi, cell_bi)[0]
    y_uni = rev_angle_embedding(y_uni.detach().numpy(), n)
    y_bi = rev_angle_embedding(y_bi.detach().numpy(), n)
    # Calculate energies
    for i in range(N_data):
        E_uni[i, n-2, :n-1] = rho_path(thetas[i, :n], phis[i, :n], y_uni[i, :n], y_uni[i, n:], dt, rho_0[i], n, 1)[2][:-1]
        E_bi[i, n-2, :n-1] = rho_path(thetas[i, :n], phis[i, :n], y_bi[i, :n], y_bi[i, n:], dt, rho_0[i], n, 1)[2][:-1]

np.save('gen/E_bi', E_bi)
np.save('gen/E_uni', E_uni)
# %%
# Visualisation
E_uni_sum = np.mean(np.cumsum(E_uni, axis=2)[:, :, -1], axis=0)
E_bi_sum = np.mean(np.cumsum(E_bi, axis=2)[:, :, -1], axis=0)
plt.scatter(N, -1 * E_uni_sum, label='Unidir. LSTM')
plt.scatter(N, -1 * E_bi_sum, label='Bidir. LSTM')
plt.legend()
plt.xlabel('$N$')
plt.ylabel('$W$')
plt.savefig('/home/fsoest/ba/phystex/img/gen50.png')
