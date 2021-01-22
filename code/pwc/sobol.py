import numpy as np
from multiproc.pwc_helpers import wrapper, get_eigen_rho, energy, state_to_angles, int_operator
from scipy.optimize import minimize
from qutip import rand_ket_haar as rkh
import matplotlib.pyplot as plt
import sobol_seq


dt = 5
N_dim = 5
N_runs = 1
N_sobol = 25

# %%
res = np.zeros((N_runs, N_sobol))
sols = np.zeros((N_runs, N_sobol, 2 * N_dim))
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
        minim = minimize(wrapper, v, args=(theta_d, phi_d, dt, rho_0, N_dim))
        res[i, k] = minim.fun
        sols[i, k] = minim.x

# %%

for i in range(N_runs):
    plt.scatter(range(len(res[i])), res[i])

np.min(res[:, :15], axis=1).mean()



res.shape
np.argmin(res, axis=1)
