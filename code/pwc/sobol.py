import numpy as np
from pwc_helpers import wrapper, get_eigen_rho, energy, state_to_angles, int_operator
from scipy.optimize import minimize
from qutip import rand_ket_haar as rkh
import matplotlib.pyplot as plt
import sobol_seq


dt = 5
N_dim = 5
N_runs = 1
N_sobol = 50

# %%
res = np.zeros((N_runs, N_sobol))
for i in range(N_runs):
    kets = np.zeros((N_dim, 2, 1), dtype=np.cdouble)
    for j in range(N_dim):
        kets[j] = rkh(2).full()
        theta_d, phi_d = state_to_angles(kets)
    phi_d[0] = np.pi*0.99
    rho_0 = get_eigen_rho(theta_d, phi_d)[0]

    vec = sobol_seq.i4_sobol_generate(2 * N_dim, N_sobol)
    vec[:, :2] *= np.pi
    vec[:, 2:] *= 2 * np.pi
    for k, v in enumerate(vec):
        res[i, k] = minimize(wrapper, v, args=(theta_d, phi_d, dt, rho_0, N_dim)).fun


np.min(res)
# %%
for i in range(N_runs):
    plt.scatter(range(len(res[i])), res[i])
# %%
opt = minimize(wrapper, vec[-1], args=(theta_d, phi_d, dt, rho_0, N_dim))

opt.fun

from multiproc.data_preprocessing import equivalent_vectors
equivalent_vectors(opt.x, 2)
opt.x
phi_d

4.9933492 - 1.8517655

phi_d

wrapper(np.array([np.pi/2, np.pi/2, 0, 6.3]), theta_d, phi_d, 50, rho_0, 2)

from N2_analyt import E_eigen
data = np.load('multi_train_data/N_2/dt_5_eigen_sobol_10_run_0.npy', allow_pickle=True)

data[5000, 2]
E_eigen(data[5000, 1], data[5000, 0], 10, data[5000, 3])
