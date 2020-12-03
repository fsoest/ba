import sys
from scipy.optimize import minimize
import numpy as np
from data_preprocessing import equivalent_vectors
from qutip import rand_ket_haar as rkh
from pwc_helpers import state_to_angles, wrapper


# Get parameters for Optimisation
args = sys.argv
if len(sys.argv) == 6:
    N_dim = int(args[1])
    N_data = int(args[2])
    dt = float(args[3])
    rho = args[4]
    run = int(args[5])
else:
    print('Parameters: N_dim, N_data, dt, rho: [haar, 0, eigen], run')
    sys.exit()

def alpha(theta, phi):
    return np.sin(theta) * np.exp(1j * phi) / 2

X = np.zeros((N_data, 2 * N_dim))
y = np.zeros((N_data, 2 * N_dim))
E = np.zeros(N_data)

if rho == 'haar':
    psis = np.zeros((N_data, 2, 1), dtype=np.complex128)
    for i in range(N_data):
        kets = np.zeros((N_dim, 2, 1), dtype=np.complex128)
        for j in range(N_dim):
            kets[j] = rkh(2).full()
        theta_d, phi_d = state_to_angles(kets)
        psi_0 = rkh(2)
        psis[i] = psi_0.full()
        rho_0 = (psi_0 * psi_0.dag()).full()
        res = minimize(wrapper, np.full(2 * N_dim, np.pi/3), args=(theta_d, phi_d, dt, rho_0, N_dim))
        X[i][:N_dim] = theta_d
        X[i][N_dim:] = phi_d
        y[i] = equivalent_vectors(res.x, N_dim)
        E[i] = res.fun
    np.save('train_data/psi_0_N_{0}_Data_{1}_dt_{2}_run_{3}'.format(N_dim, N_data, dt, run), psis)

elif rho == '0':
    zero_state = np.matrix([1, 0], dtype=np.complex128)
    rho_0 = zero_state.H @ zero_state
    for i in range(N_data):
        kets = np.zeros((N_dim, 2, 1), dtype=np.complex128)
        for j in range(N_dim):
            kets[j] = rkh(2).full()
        theta_d, phi_d = state_to_angles(kets)
        res = minimize(wrapper, np.full(2 * N_dim, np.pi/3), args=(theta_d, phi_d, dt, rho_0, N_dim))
        X[i][:N_dim] = theta_d
        X[i][N_dim:] = phi_d
        y[i] = equivalent_vectors(res.x, N_dim)
        E[i] = res.fun

elif rho == 'eigen':
    for i in range(N_data):
        kets = np.zeros((N_dim, 2, 1), dtype=np.complex128)
        for j in range(N_dim):
            kets[j] = rkh(2).full()
        theta_d, phi_d = state_to_angles(kets)
        alp = alpha(theta_d[0], phi_d[0])
        zero_state = np.matrix([np.conj(alp)/np.abs(alp), 1])
        rho_0 = zero_state.H @ zero_state
        res = minimize(wrapper, np.full(2 * N_dim, np.pi/3), args=(theta_d, phi_d, dt, rho_0, N_dim))
        X[i][:N_dim] = theta_d
        X[i][N_dim:] = phi_d
        y[i] = equivalent_vectors(res.x, N_dim)
        E[i] = res.fun

else:
    print('No rho init method found')
    sys.exit()

save_path_beg = 'train_data/N_{0}/dt_{1}/{2}/'.format(N_dim, int(dt), rho)
save_path_end = '_run_{0}'.format(run)
np.save(save_path_beg + 'X' + save_path_end, X)
np.save(save_path_beg + 'y' + save_path_end, y)
np.save(save_path_beg + 'E' + save_path_end, E)
