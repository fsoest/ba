import numpy as np
from pwc_helpers import wrapper, get_eigen_rho, energy, state_to_angles, int_operator
from scipy.optimize import minimize
from qutip import rand_ket_haar as rkh
import matplotlib.pyplot as plt
import sobol_seq


dt = 5
N_dim = 5
N_runs = 1
N_sobol = 10

def m_wrapper(x, t, p, dt, rho_0, N_dim):
    return wrapper(x, t, p, dt, rho_0, N_dim)

# %%
res = np.zeros((N_runs, N_sobol))
for i in range(N_runs):
    kets = np.zeros((N_dim, 2, 1), dtype=np.cdouble)
    for j in range(N_dim):
        kets[j] = rkh(2).full()
        theta_d, phi_d = state_to_angles(kets)

    rho_0 = np.matrix([1, 0], dtype=np.complex256).H @ np.matrix([1, 0], dtype=np.complex256)

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
np.min(res)
