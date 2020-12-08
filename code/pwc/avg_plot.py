import numpy as np
import matplotlib.pyplot as plt

run = 0
N_sobol = 10
rho = 0
dt_start = 0
dt_stop = 5

N = np.array([2])

data = np.zeros((len(N), 20, 500))
avg = np.zeros((len(N), 20))
std = np.zeros((len(N), 20))

for i, N_dim in enumerate(N):
    data[i, :, :] = np.load('multiproc/train_data/dt/vardt_N_{0}_rho_{1}/dt_{2}_{3}_E_sobol_{4}_run_{5}.npy'.format(N_dim, rho, int(dt_start), int(dt_stop), N_sobol, run))
    avg[i] = np.mean(data[i], axis=1)
    std[i] = np.std(data[i], ddof=1, axis=1)

T = np.load('multiproc/train_data/dt/vardt_N_{0}_rho_{1}/dt_{2}_{3}_times.npy'.format(N_dim, rho, int(dt_start), int(dt_stop)))

for i, n in enumerate(N):
    plt.errorbar(T, -1*avg[i]/n, yerr=std[i]/np.sqrt(500), label=n)

absa = 0.5
x = np.linspace(0, np.pi/2, 100)
plt.plot(x, 0.42 * np.sin(2 * absa * x), color='k', linestyle=':')

plt.legend(title='N')
plt.xlabel('$\Delta T$')
plt.ylabel('$\\overline{W}/N$')
# plt.savefig('/home/fsoest/ba/phystex/img/dt_dep_theor.png', dpi=300)
