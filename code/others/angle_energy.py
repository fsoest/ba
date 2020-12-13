import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from data_preprocessing import angle_embedding
# %%
def int_operator(theta, phi):
    """
    Returns partial system Hamiltonian, A * sigma_p
    """
    sigma_p = np.matrix([[0, 0], [1, 0]], dtype=np.complex128)
    sigma_m = np.matrix([[0, 1], [0, 0]], dtype=np.complex128)
    return np.sin(theta) * np.exp(1j * phi) * sigma_p / 2 + np.sin(theta) * np.exp(-1j * phi) * sigma_m / 2


def H_s(theta_d, phi_d, a, b):
    """
    """
    drive_op = int_operator(theta_d, phi_d)
    trans_op = int_operator(a, b)
    return drive_op + trans_op


def energy(theta_d, phi_d, theta_t, phi_t, dt, rho_0, N):
    E = np.zeros(N, dtype=np.complex128)
    rho = rho_0
    for i in range(N - 1):
        U = expm(-1j * dt * H_s(theta_d[i], phi_d[i], theta_t[i], phi_t[i]))
        # Time evolution of state density matrix
        rho = U @ rho @ np.matrix(U).H
        # Change in Hamiltonian due to Transducer, Drive constant
        delta_trans = np.sin(theta_t[i + 1]) * np.exp(1j * phi_t[i + 1]) - np.sin(theta_t[i]) * np.exp(1j * phi_t[i])
        dH = int_operator(theta_t[i+1], phi_t[i+1]) - int_operator(theta_t[i], phi_t[i])
        E[i] = np.trace(rho @ dH)

    return np.cumsum(E.real)[-1]


def wrapper(transducer, theta_d, phi_d, dt, rho_0, N):
    theta_t = transducer[:int(len(transducer)/2)]
    phi_t = transducer[int(len(transducer)/2):]
    return energy(theta_d, phi_d, theta_t, phi_t, dt, rho_0, N)

# %%
N = 5
dt = 1
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state

# %%
X = np.load('train_data/x_N_5_xyz.npy')
y = np.zeros((16800, 2 * N))
for i, x in enumerate(X):
    theta_d = x[:N]
    phi_d = x[N:]
    res = minimize(wrapper, np.full(2 * N, 0.5 * np.pi), args=(theta_d, phi_d, dt, rho_0, N))
    scale = res.x % (2 * np.pi)
    selection = scale[:N] > np.pi
    scale[:N][selection] -= 2 * np.pi
    scale[:N][selection] *= -1
    scale[N:][selection] += np.pi
    y[i] = scale
    if i % 1000 == 0:
        print(i)

# %%
np.save('train_data/y_N_5_ang', y)

# %%
# Train a network
input = angle(X, N)
