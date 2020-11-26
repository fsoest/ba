import numpy as np
import matplotlib.pyplot as plt

N = [2, 4, 5, 10]
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
    plt.errorbar(times, avg[i], yerr=std[i], label=n)

plt.legend(title='N')
plt.xlabel('$\Delta t$')
plt.ylabel('$-W$')

# %%

movavg = np.zeros((20, 500))
for i in range(20):
    for j in range(500):
        movavg[i, j] = np.mean(data[2, i, :j])

for i in range(20):
    plt.plot(range(500), movavg[i]-movavg[i,-1])
