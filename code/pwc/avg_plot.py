import numpy as np
import matplotlib.pyplot as plt

N = np.flip(np.array([2, 4, 5, 10]))
data = np.zeros((len(N), 20, 500))
avg = np.zeros((len(N), 20))
std = np.zeros((len(N), 20))
for i, n in enumerate(N):
    data[i, :, :] = np.load('work_N_{0}.npy'.format(n))
    avg[i] = np.mean(data[i], axis=1)
    std[i] = np.std(data[i], ddof=1, axis=1)

N_times = len(avg[0])
times = np.linspace(1e-7, 10, N_times)
for i, n in enumerate(N):
    plt.errorbar(times, -1*avg[i]/n, yerr=std[i]/np.sqrt(500), label=n)

# x = np.linspace(0, np.pi/2, 100)
# plt.plot(x, 0.4 * np.sin(x), color='k', linestyle=':')

plt.legend(title='N')
plt.xlabel('$\Delta T$')
plt.ylabel('$\\overline{W}/N$')
plt.savefig('/home/fsoest/ba/phystex/img/dt_dep_theor.png', dpi=300)
