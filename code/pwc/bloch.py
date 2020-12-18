import numpy as np
from multiproc.pwc_helpers import wrapper, get_eigen_rho
import matplotlib.pyplot as plt

dt = 5
N = 2
data = np.load('multi_train_data/N_2/dt_5_eigen_sobol_10_run_0.npy', allow_pickle=True)

fig, axes = plt.subplots(2, 1)
axes[0].set_xlabel('theta')
axes[1].set_xlabel('phi')
axes[0].set_ylabel('W')
axes[1].set_ylabel('W')

thetas = np.linsp2ace(0, np.pi, 100)
phis = np.linspace(0, 2 * np.pi, 100)

for k in range(1):
    d = data[k]

    E_theta = np.zeros(len(thetas))
    E_phi = np.zeros(len(phis))

    for i, theta in enumerate(thetas):
        E_theta[i] = -1 * wrapper(d[1], np.array([theta, 0]), d[0][2:], dt, d[3], N)

    for i, phi in enumerate(phis):
        E_phi[i] = -1 * wrapper(d[1], d[0][:2], np.array([phi, 0]), dt, d[3], N)
    phi_opt = d[0][2]

    axes[0].plot(thetas, E_theta)
    axes[0].scatter(d[0][0], -1*d[2], c='r')

    axes[1].plot(phis, E_phi)
    axes[1].scatter(d[0][2], -1*d[2], c='r')
plt.tight_layout()
# %%
import qutip
qutip.rand_herm(2, density=1)

# %%
phis = np.linspace(3.2, 3.6, 100)
thetas = np.linspace(1.7, 2.2, 100)
E = np.zeros((len(phis), len(thetas)))
for i, phi in enumerate(phis):
    for j, theta in enumerate(thetas):
        E[i, j] = -1 * wrapper(d[1], np.array([theta, 0]), np.array([phi, 0]), dt, d[3], N)
E_plot = E#np.exp(E) / np.sum(E)
plt.contourf(thetas, phis, E_plot, 100, cmap='Coolwarm')
plt.colorbar()
plt.scatter(d[0][0], d[0][2], c='r')
