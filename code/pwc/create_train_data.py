import sys
from pwc_helpers import wrapper
from scipy.optimize import minimize
import numpy as np
from data_preprocessing import equivalent_vectors
from progress_bar import printProgressBar


# Get parameters for Optimisation
args = sys.argv
if len(sys.argv) == 4:
    N_dim = int(args[1])
    N_data = int(args[2])
    dt = int(args[3])
else:
    print('Parameters: N_dim, N_data, dt')
    sys.exit()

zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state

X = np.zeros((N_data, 2 * N_dim))
y = np.zeros((N_data, 2 * N_dim))

for i in range(N_data):
    theta_d = np.random.uniform(0, np.pi, N_dim)
    phi_d = np.random.uniform(0, 2 * np.pi, N_dim)
    res = minimize(wrapper, np.full(2 * N_dim, np.pi/3), args=(theta_d, phi_d, dt, rho_0, N_dim))
    X[i][:N_dim] = theta_d
    X[i][N_dim:] = phi_d
    y[i] = equivalent_vectors(res.x, N_dim)
    printProgressBar(i, N_data)


save_path_beg = 'train_data/'
save_path_end = '_N_{0}_{1}_dt_{2}'.format(N_dim, N_data, dt)
np.save(save_path_beg + 'X' + save_path_end, X)
np.save(save_path_beg + 'y' + save_path_end, y)
