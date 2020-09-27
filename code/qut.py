import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

u_drive = np.cos(np.linspace(0, 2 * np.pi, 100))
v_drive = np.zeros(100)
w = np.sin(np.linspace(0, 2 * np.pi, 100))

theta = np.linspace(0, 2 * np.pi, 100)
phi = np.pi * np.sin(np.linspace(0, 2 * np.pi, 100))


# def bloch_to_vec(u, v, w):
#     """
#     """
#     theta = np.arccos(w)
#     phi = np.arcsin(v / np.sin(theta))
#     psi = np.cos(theta / 2) * qt.fock(2, 0) + np.exp(1j * phi) * np.sin(theta / 2) * qt.fock(2, 1)
#     return psi


def bloch_to_vec(theta, phi):
    psi = np.cos(theta / 2) * qt.fock(2, 0) + np.exp(1j * phi) * np.sin(theta / 2) * qt.fock(2, 1)
    return psi


def vec_derivative(theta, phi, dtheta, dphi):
    """
    """
    dpsi_dt = -1 * np.sin(theta / 2) * dtheta * qt.fock(2, 0) \
        + np.exp(1j * phi) * (np.cos(theta / 2) * dtheta + \
        1j * dphi * np.sin(theta / 2)) * qt.fock(2, 1)
    return dpsi_dt


# for i in range(100):
#     print(bloch_to_vec(u_drive[i], v_drive[i], w[i]))


def rho(psi):
    """
    """
    return psi * psi.dag()


s_plus = qt.sigmap()                     # Creation operator
s_minus = qt.sigmam()                    # Annihilation operator

# Interaction Hamiltonian
H_i = qt.tensor(s_plus, s_minus) + qt.tensor(s_minus, s_plus)

# DST Hamiltonian
H_dst = qt.tensor(H_i, qt.qeye(2)) + qt.tensor(qt.qeye(2), H_i)

t = np.linspace(0, 10, 100)


# %%
# Evolution of state vector
# u_d = qt.Cubic_Spline(t[0], t[-1], u_drive)
# v_d = qt.Cubic_Spline(t[0], t[-1], v_drive)
# w_d = qt.Cubic_Spline(t[0], t[-1], w)
u_d = CubicSpline(t, u_drive)
v_d = CubicSpline(t, v_drive)
w_d = CubicSpline(t, w)

theta_d = CubicSpline(t, theta)
phi_d = CubicSpline(t, phi)

# drive = [u_d, v_d, w_d]
drive = [theta_d, phi_d]
transducer = drive

def H_s(t, args):
    """
    """
    drive, transducer, H_dst = args
    # psi_d = bloch_to_vec(drive[0](t), drive[1](t), drive[2](t))
    # psi_t = bloch_to_vec(transducer[0](t), transducer[1](t), transducer[2](t))
    psi_d = bloch_to_vec(drive[0](t), drive[1](t))
    psi_t = bloch_to_vec(transducer[0](t), transducer[1](t))
    H_s = (H_dst * qt.tensor(rho(psi_d), qt.qeye(2), rho(psi_t))).ptrace(1)
    return H_s

res = qt.sesolve(H_s, qt.fock(2, 0), t, args=[drive, transducer, H_dst])

# %%
b = qt.Bloch()
b.add_states(res.states)
b.show()
# %%

def P(t, args):
    """
    """
    drive, system, transducer, H_dst = args
    psi_d = bloch_to_vec(drive[0](t), drive[1](t))#, drive[2](t))
    psi_t = bloch_to_vec(transducer[0](t), transducer[1](t))#, transducer[2](t))
    # time derivative of psi_t
    dpsi_dt = vec_derivative(transducer[0](t), transducer[1](t), transducer[0].derivative()(t), transducer[1].derivative()(t))
    bra = qt.tensor(psi_d, system[t], dpsi_dt).dag()
    ket = qt.tensor(psi_d, system[t], psi_t)
    scalar = bra * 1j * H_dst * ket
    p = 2 * scalar.full()[0].imag
    return p

args = drive, res.states, transducer, H_dst
