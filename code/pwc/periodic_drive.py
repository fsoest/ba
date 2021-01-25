import torch
import numpy as np
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding
import matplotlib.pyplot as plt
from rho_vis import exp_xyz, data_wrapper
from multiproc.pwc_helpers import wrapper, rho_path, get_eigen_rho
# %%
# %%
# Import model
N = 75
model = torch.load('models/dt_1_bi').eval()
model.N = N
dt = 1

# Create periodic drive
tau = 3 * dt
t = np.linspace(0, N-1, N)
drive_theta = np.pi/2 * (np.cos(2 * np.pi * t / tau) + 1)
drive_phi = np.pi * (np.cos(2 * np.pi * t /tau) + 1)
drive = np.concatenate((drive_theta, drive_phi))

drive_embed = angle_embedding(drive[np.newaxis], model.N)

with torch.no_grad():
    hidden, cell = model.HiddenCellTest(1)
    y_embed = model(torch.from_numpy(drive_embed), hidden, cell)[0].detach().numpy()

trans = rev_angle_embedding(y_embed, model.N)
trans.shape

# %%
num_steps = 20
rho_0 = get_eigen_rho(drive_theta[np.newaxis, 0], drive_phi[np.newaxis, 0])
# Visualisation
rho_worst_pred, rho_step_worst, E_worst_pred = rho_path(drive_theta, drive_phi, trans[0, :model.N], trans[0, model.N:], dt, rho_0, model.N, num_steps)

expec_worst_pred = exp_xyz(rho_worst_pred)
x = np.zeros(num_steps*(N-1)+1)
x[1:] = np.linspace(1/num_steps, N-1, num_steps*(N-1))
# %%
ax1 = plt.subplot(221)
ax1.plot(x, expec_worst_pred[:, 0], label='Prediction')
ax1.set_ylabel('$<\sigma_x>$')
ax1.set_ylim(-1.05, 1.05)

ax2 = plt.subplot(222)
ax2.plot(x,expec_worst_pred[:, 1], label='Prediction')
ax2.set_ylabel('$<\sigma_y>$')
ax2.set_ylim(-1.05, 1.05)


ax3 = plt.subplot(223)
ax3.plot(x, expec_worst_pred[:, 2], label='Prediction')
ax3.set_ylabel('$<\sigma_z>$')
ax3.set_ylim(-1.05, 1.05)


ax4 = plt.subplot(224)
E_plot_pred = np.zeros(N)
E_plot_real = np.zeros(N)
E_plot_pred[1:] = -1*np.cumsum(np.real(E_worst_pred[:-1]))
ax4.plot(E_plot_pred, label='Bidir. LSTM')
ax4.legend()
ax4.set_ylabel('$W$')

plt.tight_layout()
