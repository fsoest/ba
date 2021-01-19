import torch
import numpy as np
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding, angle_embedding
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
from torch.utils.data import DataLoader
from multiproc.pwc_helpers import wrapper, rho_path
import matplotlib.pyplot as plt
from rho_vis import exp_xyz, data_wrapper
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
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
# %%
model = torch.load('models/curr_dt1').eval()

hidden, cell = model.HiddenCellTest(14400)
x_test_embed = test_set.__getitem__(range(14400))['x']

with torch.no_grad():
    y_pred = model(x_test_embed, hidden, cell)

trans_pred = rev_angle_embedding(y_pred[0].detach().numpy(), N)

pred_work = np.zeros(len(trans_pred))
for i, pred in enumerate(trans_pred):
    pred_work[i] = wrapper(pred, data_test[i, 0][:N], data_test[i, 0][N:], dt, data_test[i, 3], N)

# %%
trans_real = np.zeros((len(data_test), 10))
for i, data in enumerate(data_test):
    trans_real[i] = data[1]
# Boxplots
# %%
plt.boxplot(trans_pred[:, :N])
plt.ylabel('$\\theta_{pred}$')
plt.xlabel('Transducer qubit')
# plt.savefig('/home/fsoest/ba/phystex/img/theta_pred_box.png', dpi=300)
# %%
plt.boxplot(trans_real[:, :N])
plt.ylabel('$\\theta_{opt}$')
plt.xlabel('Transducer qubit')
# plt.savefig('/home/fsoest/ba/phystex/img/theta_opt_box.png', dpi=300)
# %%
plt.boxplot(trans_pred[:, N:] % (2*np.pi))
plt.ylabel('$\phi_{pred}$')
plt.xlabel('Transducer qubit')
# plt.savefig('/home/fsoest/ba/phystex/img/phi_pred_box.png', dpi=300)
# %%
plt.boxplot(trans_real[:, N:] % (2*np.pi))
plt.ylabel('$\phi_{opt}$')
plt.xlabel('Transducer qubit')
# plt.savefig('/home/fsoest/ba/phystex/img/phi_opt_box.png', dpi=300)
# %%
plt.boxplot(np.abs((trans_real[:, N:] % (2*np.pi)) - (trans_pred[:, N:] % (2*np.pi))))
plt.ylabel('$|\phi_{opt} - \phi_{pred}|$')
plt.xlabel('Transducer qubit')
# plt.savefig('/home/fsoest/ba/phystex/img/delta_phi_box.png', dpi=300)
# %%
plt.boxplot(np.abs((trans_real[:, :N]) - (trans_pred[:, :N])))
plt.ylabel('$|\\theta_{opt} - \\theta_{pred}|$')
plt.xlabel('Transducer qubit')
# plt.savefig('/home/fsoest/ba/phystex/img/delta_theta_box.png', dpi=300)
# %%
# Steps for dt for rho trajectory
num_steps = 20
pred_work.argmax()
# Calculate trajectories
# curr_arg = 1437#pred_work.argmin()
curr_arg = 0#pred_work.argmin()#np.argsort(pred_work)[len(pred_work)//2]
rho_worst_pred, rho_step_worst, E_worst_pred = rho_path(data_test[curr_arg, 0][:N], data_test[curr_arg, 0][N:], trans_pred[curr_arg, :N], trans_pred[curr_arg, N:], dt, data_test[curr_arg, 3], N, num_steps)
rho_worst_real, rho_step_worst_real, E_worst_real = rho_path(data_test[curr_arg, 0][:N], data_test[curr_arg, 0][N:], data_test[curr_arg, 1][:N], data_test[curr_arg, 1][N:], dt, data_test[curr_arg, 3], N, num_steps)

expec_worst_pred = exp_xyz(rho_worst_pred)
expec_worst_real = exp_xyz(rho_worst_real)
x = np.zeros(num_steps*(N-1)+1)
x[1:] = np.linspace(1/num_steps, N-1, num_steps*(N-1))
# %%
ax1 = plt.subplot(221)
ax1.plot(x, expec_worst_pred[:, 0], label='Prediction')
ax1.plot(x, expec_worst_real[:, 0], label='Optimum')
ax1.set_ylabel('$<\sigma_x>$')
ax1.set_ylim(-1.05, 1.05)

ax2 = plt.subplot(222)
ax2.plot(x,expec_worst_pred[:, 1], label='Prediction')
ax2.plot(x,expec_worst_real[:, 1], label='Optimum')
ax2.set_ylabel('$<\sigma_y>$')
ax2.set_ylim(-1.05, 1.05)


ax3 = plt.subplot(223)
ax3.plot(x, expec_worst_pred[:, 2], label='Prediction')
ax3.plot(x, expec_worst_real[:, 2], label='Optimum')
ax3.set_ylabel('$<\sigma_z>$')
ax3.set_ylim(-1.05, 1.05)


ax4 = plt.subplot(224)
E_plot_pred = np.zeros(N)
E_plot_real = np.zeros(N)
E_plot_pred[1:] = -1*np.cumsum(np.real(E_worst_pred[:-1]))
E_plot_real[1:] = -1*np.cumsum(np.real(E_worst_real[:-1]))
ax4.plot(E_plot_pred, label='Bidir. LSTM')
ax4.plot(E_plot_real, label='Optimum')
ax4.legend()
ax4.set_ylabel('$W$')

plt.tight_layout()
# plt.savefig('/home/fsoest/ba/phystex/img/path_10553.png', dpi=300)

# %%
# Bloch sphere Hamiltonian Plot
def bloch_hamiltonian(data, dt, N, trans):
    rhos_pred, rho_step, E_pred = rho_path(data[0][:N], data[0][N:], trans[:N], trans[N:], dt, data[3], N, 1)
    rhos_opt, rho_step, E_opt = rho_path(data[0][:N], data[0][N:], data[1][:N], data[1][N:], dt, data[3], N, 1)

    rho_opt_vec = np.zeros((N, 3))
    h_sys_opt = np.zeros((N, 3))

    rho_pred_vec = np.zeros((N, 3))
    h_sys_pred = np.zeros((N, 3))

    h_drive = np.zeros((N, 3))

    for i in range(N):
        rho_opt_vec[i] = np.array([2 * np.real(rhos_opt[i][0, 1]), 2 * np.imag(rhos_opt[i][1, 0]), rhos_opt[i][0, 0] - rhos_opt[i][1, 1]])
        h_sys_opt[i] = np.sin(data[1][i]) *  np.array([np.cos(data[1][i + N]), np.sin(data[1][i + N]), 0])

        rho_pred_vec[i] = np.array([2 * np.real(rhos_pred[i][0, 1]), 2 * np.imag(rhos_pred[i][1, 0]), rhos_pred[i][0, 0] - rhos_pred[i][1, 1]])
        h_sys_pred[i] = np.sin(trans[i]) *  np.array([np.cos(trans[i + N]), np.sin(trans[i + N]), 0])

        h_drive[i] = np.sin(data[0][i]) *  np.array([np.cos(data[0][i + N]), np.sin(data[0][i + N]), 0])

    # fig, axs = plt.subplots(nrows=1, ncols=(N-1))
    fig = plt.figure(figsize=(15, 8))
    for j in range(N-1):
        ax = fig.add_subplot(2, 4, j + 1, projection='3d')
        b = cBloch(fig=fig, axes=ax)
        b.add_points(rho_opt_vec[j], colors='b')
        b.add_points(rho_opt_vec[j+1], colors='k')
        b.add_vectors(h_drive[j], colors='r')
        b.add_vectors(h_sys_opt[j], colors='y')
        b.add_vectors(h_sys_opt[j+1], colors='g')
        b.render(fig=fig, axes=ax)

        ax = fig.add_subplot(2, 4, j + 5, projection='3d')
        b = cBloch(fig=fig, axes=ax)
        b.add_points(rho_pred_vec[j], colors='b')
        b.add_points(rho_pred_vec[j+1], colors='k')
        b.add_vectors(h_drive[j], colors='r')
        b.add_vectors(h_sys_pred[j], colors='y')
        b.add_vectors(h_sys_pred[j+1], colors='g')
        b.render(fig=fig, axes=ax)
    # plt.show()
    # plt.savefig('/home/fsoest/ba/phystex/img/bloch_10553.png', dpi=300)

bloch_hamiltonian(data_test[curr_arg], dt, N, trans_pred[curr_arg])

# %%
data_comp = np.load('multiproc/train_data/N_5/dt_1_eigen_sobol_45_run_0.npy', allow_pickle=True)
x_embed = torch.from_numpy(angle_embedding(data_comp[0, 0, np.newaxis], 5))
hidden, cell = model.HiddenCellTest(1)
y_pred = model(x_embed, hidden, cell)
trans_1 = rev_angle_embedding(y_pred[0].detach().numpy(), N)
bloch_hamiltonian(data_comp[0], dt, N, trans_1[0])
plt.savefig('/home/fsoest/ba/phystex/img/bloch_comp_1.png', dpi=300)
data_comp
wrapper(trans_1[0], data_comp[0, 0][:N], data_comp[0, 0][N:], dt, data_comp[0, 3], N)

# %%
curr_arg = pred_work.argmin()
# Look at variation in drive
inp = np.copy(data_test[curr_arg, 0])
emb_inp = torch.from_numpy(angle_embedding(inp[np.newaxis], 5))
hidden, cell = model.HiddenCellTest(1)
emb_out = model(emb_inp, hidden, cell)[0].detach().numpy()
out = rev_angle_embedding(emb_out, 5)[0]
# Create input that is different in one qubit
inp_1 = np.copy(inp)
inp_1[1] = np.pi/2
inp_1[6] += 1.41
emb_inp_1 = torch.from_numpy(angle_embedding(inp_1[np.newaxis], 5))
hidden, cell = model.HiddenCellTest(1)
emb_out_1 = model(emb_inp_1, hidden, cell)[0].detach().numpy()
out_1 = rev_angle_embedding(emb_out_1, 5)[0]


# Calculate dynamics

rhos_0, rho_0, E_0 = rho_path(inp[:N], inp[N:], out[:N], out[N:], dt, data_test[curr_arg, 3], N, 1)
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

# %%
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
