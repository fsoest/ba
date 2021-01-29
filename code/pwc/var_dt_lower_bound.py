import numpy as np
import matplotlib.pyplot as plt


N = [2, 3, 4, 5, 9, 10]
t = np.load('lower_dt/vardt_N_5_rho_eigen_dt_0_5_times.npy')

for n in N:
    e = np.load('lower_dt/vardt_N_{0}_rho_eigen_dt_0_5_run_0_E.npy'.format(n))
    std = np.std(e, axis=1, ddof=1)
    plt.errorbar(t, -1*np.mean(e, axis=1)/(n-1), yerr=std/(n-1), label=n)
plt.legend(title='N')
plt.savefig('/home/fsoest/ba/phystex/img/vardt_lower_bound.png', dpi=300)
