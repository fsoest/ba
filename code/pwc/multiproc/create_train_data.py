import sys
from scipy.optimize import minimize
import numpy as np
from data_preprocessing import equivalent_vectors
from qutip import rand_ket_haar as rkh
from pwc_helpers import state_to_angles, wrapper, get_eigen_rho
import sobol_seq


def make_rho_0(rho, theta_d_0, phi_d_0):
    """
    Creates correct initial system state, depending on method
    """
    if rho == 'haar':
        psi_0 = rkh(2)
        rho_0 = (psi_0 * psi_0.dag()).full()
        return rho_0
    elif rho == '0':
        zero_state = np.matrix([1, 0], dtype=np.complex128)
        rho_0 = zero_state.H @ zero_state
        return rho_0
    elif rho == 'eigen':
        # get_eigen_rho returns array of rho_0, therefore [0]
        rho_0 = get_eigen_rho(np.array([theta_d_0]), np.array([phi_d_0]))[0]
        return rho_0

def create_data(N_dim, dt, rho, N_sobol, seed):
    # Seed
    np.random.seed(seed)

    # Initial conditions from sobol
    vec = sobol_seq.i4_sobol_generate(2 * N_dim, N_sobol)
    # Rescale sobol sequences to pi/2pi
    vec[:, :N_dim] *= np.pi
    vec[:, N_dim:] *= 2 * np.pi
    # Arrays to save data
    X = np.zeros(2 * N_dim)
    y = np.zeros(2 * N_dim)
    E = 0

    # Create uniform drives
    kets = np.zeros((N_dim, 2, 1), dtype=np.complex128)
    for j in range(N_dim):
        kets[j] = rkh(2).full()
    # Angles from state vectors
    theta_d, phi_d = state_to_angles(kets)
    rho_0 = make_rho_0(rho, theta_d[0], phi_d[0])

        # Iterate over sobol sequence
    for v in vec:
        res = minimize(wrapper, v, args=(theta_d, phi_d, dt, rho_0, N_dim))
        # If result with current sobol sequence is better than previous best, update
        if res.fun < E:
            E = res.fun
            y = equivalent_vectors(res.x, N_dim)
    X[:N_dim] = theta_d
    X[N_dim:] = phi_d

    return np.array([X, y, E, rho_0])

if __name__ == '__main__':
    args = sys.argv
    if len(sys.argv) == 6:
        N_dim = int(args[1])
        dt = float(args[2])
        rho = args[3]
        N_sobol = int(args[4])
        seed = int(args[5])
    else:
        print('Parameters: N_dim, dt, rho: [haar, 0, eigen], N_sobol, seed')
        sys.exit()
    create_data(N_dim, dt, rho, N_sobol, seed)
