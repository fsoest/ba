from progress_bar import printProgressBar
from pwc_helpers import wrapper
from scipy.optimize import minimize
import numpy as np
import sys

args = sys.argv
if len(sys.argv) == 4:
    N_dim = int(args[1])
    N_data = int(args[2])
    N_times = int(args[3])
else:
    print('Parameters: N_dim N_data N_times')
    sys.exit()

# %%

zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state
times = np.linspace(1e-7, 10, N_times)

# %%
E = np.zeros((N_times, N_data))
theta_d = np.random.uniform(0, np.pi, (N_data, N_dim))
phi_d = np.random.uniform(0, 2 * np.pi, (N_data, N_dim))
# %%
for i, dt in enumerate(times):
    for p in range(N_data):
        res = minimize(wrapper, np.full(2 * N_dim, 1e-3 * np.pi), args=(theta_d[p], phi_d[p], dt, rho_0, N_dim))
        E[i, p] = res.fun
        printProgressBar(p, N_data, suffix='Time {0}'.format(dt))

np.save('work_N_{0}'.format(N_dim), E)
