import numpy as np
import matplotlib.pyplot as plt

run = 0
N_sobol = 10
rho = 0
dt_start = 0
dt_stop = 5

N = np.array([2, 3, 4, 5, 9, 10])

data = np.zeros((len(N), 20, 500))
avg = np.zeros((len(N), 20))
std = np.zeros((len(N), 20))

for i, N_dim in enumerate(N):
    data[i, :, :] = np.load('multi_train_data/dt/vardt_N_{0}_rho_{1}/dt_{2}_{3}_E_sobol_{4}_run_{5}.npy'.format(N_dim, rho, int(dt_start), int(dt_stop), N_sobol, run))
    avg[i] = np.mean(data[i], axis=1)
    std[i] = np.std(data[i], ddof=1, axis=1)


T = np.load('multi_train_data/dt/vardt_N_{0}_rho_{1}/dt_{2}_{3}_times.npy'.format(N_dim, rho, int(dt_start), int(dt_stop)))

for i, n in enumerate(N):
    plt.errorbar(T, -1*avg[i]/(n-1), yerr=std[i]/np.sqrt(500), label=n)
    # plt.errorbar(T, -1*avg_eigen[i]/n, yerr=std_eigen[i]/np.sqrt(500), label='eigen')

plt.legend(title='N')
plt.xlabel('$\Delta T$')
plt.ylabel('$\\overline{W}/(N-1)$')
# plt.hlines(0.5, 0, 5)
plt.savefig('/home/fsoest/ba/phystex/img/dt_0.png', dpi=300)
# %%
run = 0
N_sobol = 10
rho = 'eigen'
dt_start = 0
dt_stop = 5

N = np.array([2, 3, 4, 5, 9, 10])

data_eigen = np.zeros((len(N), 20, 500))
avg_eigen = np.zeros((len(N), 20))
std_eigen = np.zeros((len(N), 20))

for i, N_dim in enumerate(N):
    data_eigen[i, :, :] = np.load('multi_train_data/dt/vardt_N_{0}_rho_{1}/dt_{2}_{3}_E_sobol_{4}_run_{5}.npy'.format(N_dim, rho, int(dt_start), int(dt_stop), N_sobol, run))
    avg_eigen[i] = np.mean(data_eigen[i], axis=1)
    std_eigen[i] = np.std(data_eigen[i], ddof=1, axis=1)

T = np.load('multi_train_data/dt/vardt_N_{0}_rho_{1}/dt_{2}_{3}_times.npy'.format(N_dim, rho, int(dt_start), int(dt_stop)))

for i, n in enumerate(N):
    plt.errorbar(T, -1*avg_eigen[i]/(n-1), yerr=std_eigen[i]/np.sqrt(500), label=n)

plt.legend(title='N')
plt.xlabel('$\Delta T$')
plt.ylabel('$\\overline{W}/(N-1)$')
plt.savefig('/home/fsoest/ba/phystex/img/dt_eigen.png', dpi=300)
