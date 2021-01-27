import torch
from dataset import WorkDataset
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
import numpy as np
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding
from rho_vis import exp_xyz, data_wrapper, rho_path
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
test_set = WorkDataset(data_test, N, 'lstm')

# %%
# Import models
uni = torch.load('models/dt_1_uni').eval()
bi = torch.load('models/dt_1_bi').eval()
# %%
with torch.no_grad():
    uni_pred = uni.get_work_array(data_test, dt)
    bi_pred = bi.get_work_array(data_test, dt)

delta = uni_pred - bi_pred
plt.hist(delta)

E_uni = np.zeros((len(data_test), 5))
E_bi = np.zeros((len(data_test), 5))
E_opt = np.zeros((len(data_test), 5))
# Calculate predictions of models
with torch.no_grad():
    uni_hidden, uni_cell = uni.HiddenCellTest(len(data_test))
    d = test_set.__getitem__(range(len(data_test)))
    uni_pred = uni(d['x'], uni_hidden, uni_cell)[0]
    bi_hidden, bi_cell = bi.HiddenCellTest(len(data_test))
    bi_pred = bi(d['x'], bi_hidden, bi_cell)[0]

uni_trans = rev_angle_embedding(uni_pred, N)
bi_trans = rev_angle_embedding(bi_pred, N)

for i, data in enumerate(data_test):
    E_uni[i] = rho_path(data[0][:N], data[0][N:], uni_trans[i, :N], uni_trans[i, N:], dt, data[3], N, 1)[2]
    E_bi[i] = rho_path(data[0][:N], data[0][N:], bi_trans[i, :N], bi_trans[i, N:], dt, data[3], N, 1)[2]
    E_opt[i] = rho_path(data[0][:N], data[0][N:], data[1][:N], data[1][N:], dt, data[3], N, 1)[2]


plt.scatter(range(5), -1 * np.mean(E_uni, axis=0))
# %%
plt.hist(-1*E_uni.min(axis=1), bins=50, label='Unidir. LSTM')
plt.hist(-1*E_bi.min(axis=1), bins=50, label='Bidir. LSTM')
plt.hist(-1*E_opt.min(axis=1), bins=50, label='Opt')
plt.xlabel('$dW_{max}$')
plt.legend()

plt.hist(E_uni[:-1].argmin(axis=1))


# %%
plt.scatter(E_opt.min(axis=1), -1*E_opt.cumsum(axis=1)[:, -1], alpha=0.02)
# %%
# Plots
curr_arg = 5922
inp = np.copy(data_test[curr_arg, 0])
emb_inp = torch.from_numpy(angle_embedding(inp[np.newaxis], N))
hidden, cell = bi.HiddenCellTest(1)
emb_out = bi(emb_inp, hidden, cell)[0].detach().numpy()
out = rev_angle_embedding(emb_out, 5)[0]
rhos_0, rho_0, E_0 = rho_path(inp[:N], inp[N:], out[:N], out[N:], dt, data_test[curr_arg, 3], N, 1)

inp_1 = np.copy(inp)
emb_inp_1 = torch.from_numpy(angle_embedding(inp_1[np.newaxis], N))
hidden, cell = uni.HiddenCellTest(1)
emb_out_1 = uni(emb_inp_1, hidden, cell)[0].detach().numpy()
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



# %%
curr_arg = np.argsort(delta)[len(delta) //2] + 500
curr_arg
num_steps = 20

inp = np.copy(data_test[curr_arg, 0])
emb_inp = torch.from_numpy(angle_embedding(inp[np.newaxis], N))
hidden, cell = bi.HiddenCellTest(1)
emb_out = bi(emb_inp, hidden, cell)[0].detach().numpy()
out = rev_angle_embedding(emb_out, 5)[0]
rhos_0, rho_0, E_0 = rho_path(inp[:N], inp[N:], out[:N], out[N:], dt, data_test[curr_arg, 3], N, 1)
inp_1 = np.copy(inp)
emb_inp_1 = torch.from_numpy(angle_embedding(inp_1[np.newaxis], N))
hidden, cell = uni.HiddenCellTest(1)
emb_out_1 = uni(emb_inp_1, hidden, cell)[0].detach().numpy()
out_1 = rev_angle_embedding(emb_out_1, 5)[0]


# Calculate trajectories
# curr_arg = 1437#pred_work.argmin()
#pred_work.argmax()#np.argsort(pred_work)[len(pred_work)//2]
rho_bi_pred, rho_step_bi, E_bi_pred = rho_path(inp[:N], inp[N:], out[:N], out[N:], dt, data_test[curr_arg, 3], N, num_steps)

rho_uni_pred, rho_step_bi, E_uni_pred = rho_path(inp_1[:N], inp_1[N:], out_1[:N], out_1[N:], dt, data_test[curr_arg, 3], N, num_steps)

rho_bi_real, rho_step_bi_real, E_bi_real = rho_path(data_test[curr_arg, 0][:N], data_test[curr_arg, 0][N:], data_test[curr_arg, 1][:N], data_test[curr_arg, 1][N:], dt, data_test[curr_arg, 3], N, num_steps)


expec_bi_pred = exp_xyz(rho_bi_pred)
expec_pred = exp_xyz(rho_bi_real)
expec_uni_pred = exp_xyz(rho_uni_pred)
x = np.zeros(num_steps*(N-1)+1)
x[1:] = np.linspace(1/num_steps, N-1, num_steps*(N-1))
# %%
ax1 = plt.subplot(221)
ax1.plot(x, expec_bi_pred[:, 0], label='Bidir. LSTM')
ax1.plot(x, expec_uni_pred[:, 0], label='Unidir. LSTM')
ax1.plot(x, expec_pred[:, 0], label='Optimum')
ax1.set_ylabel('$<\sigma_x>$')
ax1.set_ylim(-1.05, 1.05)

ax2 = plt.subplot(222)
ax2.plot(x,expec_bi_pred[:, 1], label='Bidir. LSTM')
ax2.plot(x, expec_uni_pred[:, 1], label='Unidir. LSTM')
ax2.plot(x,expec_pred[:, 1], label='Optimum')

ax2.set_ylabel('$<\sigma_y>$')
ax2.set_ylim(-1.05, 1.05)


ax3 = plt.subplot(223)
ax3.plot(x, expec_bi_pred[:, 2], label='Bidir. LSTM')
ax3.plot(x, expec_uni_pred[:, 2], label='Unidir. LSTM')
ax3.plot(x, expec_pred[:, 2], label='Optimum')
ax3.set_ylabel('$<\sigma_z>$')
ax3.set_ylim(-1.05, 1.05)


ax4 = plt.subplot(224)
E_plot_pred = np.zeros(N)
E_plot_real = np.zeros(N)
E_plot_uni = np.zeros(N)
E_plot_pred[1:] = -1*np.cumsum(np.real(E_bi_pred[:-1]))
E_plot_real[1:] = -1*np.cumsum(np.real(E_bi_real[:-1]))
E_plot_uni[1:] = -1*np.cumsum(np.real(E_uni_pred[:-1]))
ax4.plot(E_plot_pred, label='Bidir. LSTM')
ax4.plot(E_plot_uni, label='Unidir. LSTM')
ax4.plot(E_plot_real, label='Optimum')
ax4.legend()
ax4.set_ylabel('$W$')

plt.tight_layout()
