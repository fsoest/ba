import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial

u_drive = np.sin(np.linspace(0, 2 * np.pi, 21))
v_drive = np.cos(np.linspace(0, 2 * np.pi, 21))
w = np.zeros(21)


def bloch_to_vec(u, v, w):
    """
    """
    theta = np.arccos(w)
    phi = np.arcsin(v / np.sin(theta))
    psi = np.cos(theta / 2) * qt.fock(2, 0) + np.exp(1j * phi) * np.sin(theta / 2) * qt.fock(2, 1)
    return psi


def rho(psi):
    """
    """
    return psi * psi.dag()


s_plus = qt.create(2)                     # Creation operator
s_minus = s_plus.dag()                    # Annihilation operator

# Interaction Hamiltonian
H_i = qt.tensor(s_plus, s_minus) + qt.tensor(s_minus, s_plus)

# DST Hamiltonian
H_dst = qt.tensor(H_i, qt.qeye(2)) + qt.tensor(qt.qeye(2), H_i)

# H_s = (H_dst * qt.tensor(rho(psi_d(3)), qt.qeye(2), rho(psi_d(np.pi)))).ptrace(1)
# H_s

t = np.linspace(0, 10, 500)


# %%
# Evolution of state vector

u_d = qt.Cubic_Spline(t[0], t[-1], u_drive)
v_d = qt.Cubic_Spline(t[0], t[-1], v_drive)
w_d = qt.Cubic_Spline(t[0], t[-1], w)
qdrive = [u_d, v_d, w_d]
transducer = drive


def H_s(t, drive=qdrive, transducer=qdrive, H_dst=H_dst):
    """
    """
    psi_d = bloch_to_vec(drive[0](t), drive[1](t), drive[2](t))
    psi_t = bloch_to_vec(transducer[0](t), transducer[1](t), transducer[2](t))
    H_s = (H_dst * qt.tensor(rho(psi_d), qt.qeye(2), rho(psi_t))).ptrace(1)
    return H_s


H_evo = partial(H_s, drive=drive, transducer=transducer, H_dst=H_dst)


args = {
    "drive": drive,
    "transducer": transducer,
    "H_dst": H_dst
    }


# qt.sesolve(H_s, qt.fock(2, 0), t, args=args)

qt.sesolve(H_s, qt.fock(2, 0), t)
