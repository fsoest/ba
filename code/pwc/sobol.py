import numpy as np
from multiproc.pwc_helpers import wrapper, get_eigen_rho, energy, state_to_angles, int_operator, rho_path
from scipy.optimize import minimize
from qutip import rand_ket_haar as rkh
import matplotlib.pyplot as plt
import sobol_seq


dt = 1
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
    rho_0 = get_eigen_rho(theta_d, phi_d)[0]

    vec = sobol_seq.i4_sobol_generate(2 * N_dim, N_sobol)
    vec[:, :2] *= np.pi
    vec[:, 2:] *= 2 * np.pi
    for k, v in enumerate(vec):
        minim = minimize(wrapper, v, args=(theta_d, phi_d, dt, rho_0, N_dim))
        res[i, k] = minim.fun
        sols[i, k] = minim.x
11res.argmin()
sols.shape
rho_path(theta_d, phi_d, sols[0, 11, :5], sols[0, 11, 5:], dt, rho_0, 5, 1)

# %%
for i in range(N_runs):
    plt.scatter(range(len(res[i])), res[i])

np.mean(res)

# %%

res.shape
np.argmin(res, axis=1)
# %%
minim = minimize(wrapper, v, args=(np.full(N_dim, np.pi/2), np.full(N_dim, 0.4), dt, rho_0, N_dim))

# %%
N_sobol = 10
res = np.zeros(N_sobol)
t_d = np.full(3, np.pi/2)
p_d = np.array([0, 1/3, 3/4])*np.pi*2
rho_0 = get_eigen_rho(t_d, p_d)[0]
vec = sobol_seq.i4_sobol_generate(2 * 3, N_sobol)
vec[:, :2] *= np.pi
vec[:, 2:] *= 2 * np.pi
for k, v in enumerate(vec):
    minim = minimize(wrapper, v, args=(t_d, p_d, 1, rho_0, 3))
    res[k] = minim.fun

plt.hist(res)
# %%
from scipy.linalg import expm


wrapper(np.array([np.pi/3, np.pi/3, np.pi/2, 0, np.pi, arg+np.pi]), t_d, p_d, 1, rho_0, 3)

sx = np.matrix([[0, 1], [1, 0]])
sy = np.matrix([[0, -1j], [1j, 0]])

H1 = np.array([[0, np.exp(-1j*p_d[1])/2], [np.exp(1j*p_d[1])/2, 0]]) - sx/2
U1 = expm(-1j * H1)

rho1 = U1 @ rho_0 @ np.matrix(U1).H
rho1
x = np.real(np.trace(rho1 @ sx))
y = np.real(np.trace(rho1 @ sy))
x
y
arg = np.arctan2(y, x)

rho_path(t_d, p_d, np.array([np.pi/2, np.pi/2, np.pi/2]), np.array([0, np.pi, arg+np.pi]), 1, rho_0, 3, 1)

np.trace(rho1 @ (-1*sx/2 + np.array([[0, np.exp(-1j*arg)/2], [np.exp(1j*arg)/2, 0]])))



# %%
