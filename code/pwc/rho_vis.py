from multiproc.pwc_helpers import rho_path
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


def data_wrapper(data, dt, steps):
    N = int(len(data[0]) / 2)
    theta_d = data[0][:N]
    phi_d = data[0][N:]
    theta_t = data[1][:N]
    phi_t = data[1][N:]
    rho_0 = data[3]
    return rho_path(theta_d, phi_d, theta_t, phi_t, dt, rho_0, N, steps)


def exp_xyz(rhos):
    exp = np.zeros((len(rhos), 3))
    for i in range(len(rhos)):
        exp[i, 0] = np.real(np.trace(qt.sigmax().full() @ rhos[i]))
        exp[i, 1] = np.real(np.trace(qt.sigmay().full() @ rhos[i]))
        exp[i, 2] = np.real(np.trace(qt.sigmaz().full() @ rhos[i]))
    return exp

# %%

dt1 = np.load('multi_train_data/N_5/dt_1_eigen_sobol_10_run_0.npy', allow_pickle=True)
dt5 = np.load('multi_train_data/N_5/dt_5_eigen_sobol_10_run_0.npy', allow_pickle=True)
dt10 = np.load('multi_train_data/N_5/dt_10_eigen_sobol_10_run_0.npy', allow_pickle=True)

# %%
inx = 0
path1, step1, e1 = data_wrapper(dt1[inx], 1, 50)
path5, step5, e5 = data_wrapper(dt5[inx], 5, 50)
path10, step10, e10 = data_wrapper(dt10[inx], 10, 50)

exp_1 = exp_xyz(path1)
exp_5 = exp_xyz(path5)
exp_10 = exp_xyz(path10)

exp5 = exp_xyz(step5)
# %%
plt.plot(range(len(e1)), -np.real(e1), label='1')
plt.plot(range(len(e5)), -np.real(e5), label='5')
plt.plot(range(len(e10)), -np.real(e10), label='10')
plt.legend(title='$\Delta T$')

# %%
a = ['x', 'y', 'z']

for i, dir in enumerate(a):
    plt.plot(range(len(exp_1)), exp_1[:, i], label='1 {0}'.format(dir))
    plt.plot(range(len(exp_5)), exp_5[:, i], label='5 {0}'.format(dir))
# for i, dir in enumerate(a):
#     plt.plot(range(len(exp_10)), exp_10[:, i], label='10 {0}'.format(dir))

# plt.plot(np.linspace(0, 5, 201), exp_1[:, 2])
# plt.plot(np.linspace(0, 5, 201), exp_5[:, 2])

# plt.plot(np.linspace(0, 5, ))

plt.legend()
plt.xlabel('t')
plt.ylabel('$<\sigma_i>$')
