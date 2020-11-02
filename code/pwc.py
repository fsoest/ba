import numpy as np
import qutip as qt
#from helpers import rho, bloch_to_vec, vec_derivative
from scipy.optimize import approx_fprime
from scipy.linalg import expm
import matplotlib.pyplot as plt
from adam import ADAM


def int_operator(theta, phi):
    """
    """
    return np.sin(theta) * np.exp(1j * phi) * np.matrix([[0, 1], [0, 0]], dtype=np.complex128)
    # return np.sin(theta) * np.exp(1j * phi) * qt.sigmap() / 2


def H_s(theta_d, phi_d, theta_t, phi_t):
    """
    """
    drive_op = int_operator(theta_d, phi_d)
    trans_op = int_operator(theta_t, phi_t)
    return drive_op + drive_op.getH() + trans_op + trans_op.getH()


def energy(theta_d, phi_d, theta_t, phi_t, dt, rho_0):
    H_A = np.zeros((N, 2, 2), dtype=np.complex128)
    H_B = np.zeros((N, 2, 2), dtype=np.complex128)
    A = np.zeros((N, 2, 2), dtype=np.complex128)
    B = np.zeros((N, 2, 2), dtype=np.complex128)
    E = np.zeros(N, dtype=np.complex128)
    rho = np.zeros((N, 2, 2), dtype=np.complex128)
    rho[0] = rho_0
    H_A[-1] = H_s(theta_d[-1], phi_d[-1], theta_t[-1], phi_t[-1])
    for i in range(N - 1):
        H_A[i] = H_s(theta_d[i], phi_d[i], theta_t[i], phi_t[i])
        H_B[i] = H_s(theta_d[i + 1], phi_d[i + 1], theta_t[i], phi_t[i])
        A[i] = expm(-1j * dt / 2 * H_A[i])
        B[i] = expm(-1j * dt / 2 * H_B[i])
        rho[i + 1] = B[i] @ A[i] @ rho[i] @ np.matrix(A[i]).H @ np.matrix(B[i]).H
        E[i] = np.trace(rho[i + 1] @ (H_A[i + 1] - H_B[i]))

    return np.cumsum(E.real)[-1]


N = 50
T = 10
theta_d = np.pi * np.sin(2 * np.pi * np.linspace(0, N, N) / T)
phi_d = np.full(N, np.pi)
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state
dt = 1

def wrapper(transducer, theta_d, phi_d, dt, rho_0):
    theta_t = transducer[:int(len(transducer)/2)]
    phi_t = transducer[int(len(transducer)/2):]
    return energy(theta_d, phi_d, theta_t, phi_t, dt, rho_0)

transducer = np.full(2*N, 1.0)


eps = np.sqrt(np.finfo(float).eps)

# %%
steps = 100
learn = np.zeros(steps)
for i in range(steps):
    transducer += 3e-3 * approx_fprime(transducer, wrapper, eps, theta_d, phi_d, dt, rho_0)
    learn[i] = wrapper(transducer, theta_d, phi_d, dt, rho_0)
plt.plot(range(steps), learn)
# %%
import matplotlib.pyplot as plt
plt.plot(range(N), transducer[N:])
plt.plot(range(N), transducer[:N])

# %%
args = theta_d, phi_d, dt, rho_0
theta_opt = ADAM(wrapper, theta_opt, 100, args=args)

wrapper(theta_opt, theta_d, phi_d, dt, rho_0)
