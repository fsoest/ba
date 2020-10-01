import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N = 30

def bloch_to_vec(theta, phi):
    """
    """
    psi = np.cos(theta / 2) * qt.fock(2, 0) + np.exp(1j * phi) * np.sin(theta / 2) * qt.fock(2, 1)
    return psi


def vec_derivative(theta, phi, dtheta, dphi):
    """
    """
    dpsi_dt = -1 * np.sin(theta / 2) * dtheta * qt.fock(2, 0) \
        + np.exp(1j * phi) * (np.cos(theta / 2) * dtheta + \
        1j * dphi * np.sin(theta / 2)) * qt.fock(2, 1)
    return dpsi_dt


def rho(psi):
    """
    """
    return psi * psi.dag()


def rhs(t, y, t_arr, t_drive, p_drive, t_transducer, p_transducer, H_dst):
    """
    t: time
    y: array, [E, |psi_s>]
    """
    # Spline functions
    theta_d = CubicSpline(t_arr, t_drive)
    phi_d = CubicSpline(t_arr, p_drive)
    theta_t = CubicSpline(t_arr, t_transducer)
    phi_t = CubicSpline(t_arr, p_transducer)

    # System hamiltonian
    psi_d = bloch_to_vec(theta_d(t), phi_d(t))  # Qobj
    psi_s = y[1:]                         # Array!!!
    psi_t = bloch_to_vec(theta_t(t), phi_t(t))  # Qobj
    H_s = (H_dst * qt.tensor(rho(psi_d), qt.qeye(2), rho(psi_t))).ptrace(1)

    # Evolution
    dpsi_s_dt = -1j * H_s.full() @ psi_s

    # Time derivative of transducer state
    dpsi_t_dt = vec_derivative(theta_t(t), phi_t(t), theta_t.derivative()(t), \
        phi_t.derivative()(t))

    # Overlap
    bra = qt.tensor(psi_d, qt.Qobj(psi_s), dpsi_t_dt).dag()
    ket = qt.tensor(psi_d, qt.Qobj(psi_s), psi_t)
    scalar = bra * -1j * H_dst * ket
    p = -2 * scalar.full()[0].imag
    res = np.full(3, p, dtype=np.complex)
    res[1:] = dpsi_s_dt
    return res


# %%
# Interaction Hamiltonian
H_i = qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap())
# DST Hamiltonian
H_dst = qt.tensor(H_i, qt.qeye(2)) + qt.tensor(qt.qeye(2), H_i)

theta = np.linspace(0, 2 * np.pi, N)
phi = np.pi * np.sin(np.linspace(0, 2 * np.pi, N))

t_arr = np.linspace(0, 10, N)

# %%
trans_t = np.random.uniform(0, 2 * np.pi, N)
trans_p = np.random.uniform(0, 2 * np.pi, N)


args = (t_arr, theta, phi, trans_t, trans_p, H_dst)

t_span = (0, 10)

# rhs(0.5, np.array([0, 1+0j, 0+0j]), *args)

E = solve_ivp(rhs, t_span, np.array([0, 1/np.sqrt(2) + 0j, 1/np.sqrt(2) + 0j]), args=args)

# %%

states = []
b = qt.Bloch()

for i in range(len(E.y[1:].T)):
    b.add_states(qt.Qobj(E.y[1:].T[i]))
b.show()

# %%
plt.plot(E.t, E.y[0].real)
# %%



# Was fordern? Anfang = End?
# Optimierung
 # RL - Annealing
 # Gradient ascent
 # Diskrete phi, theta
 #
