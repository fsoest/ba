import numpy as np
import qutip as qt


def bloch_to_vec(theta, phi):
    """
    """
    psi = np.cos(theta / 2) * qt.fock(2, 0) + np.exp(1j * phi) * np.sin(theta / 2) * qt.fock(2, 1)
    return psi


def vec_derivative(theta, phi, dtheta, dphi):
    """
    """
    dpsi_dt = -1 * np.sin(theta / 2) * dtheta / 2 * qt.fock(2, 0) \
        + np.exp(1j * phi) * (np.cos(theta / 2) * dtheta / 2 + \
        1j * dphi * np.sin(theta / 2)) * qt.fock(2, 1)
    return dpsi_dt


def rho(psi):
    """
    Returns pure-state density matrix from psi
    """
    return psi * psi.dag()


def rhs(t, y, theta_d, phi_d, theta_t, phi_t, H_dst):
    """

    t: time
    y: array, [E, |psi_s>]
    """

    # System hamiltonian
    psi_d = bloch_to_vec(theta_d(t), phi_d(t))  # Qobj
    psi_s = y[1:]                               # Array!!!
    psi_t = bloch_to_vec(theta_t(t), phi_t(t))  # Qobj
    # Reduced system Hamiltonian
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


def t_spline(t_mdp, alpha):
    """
    Creates compatible Spline array from t_mdp array
    """
    dt = t_mdp[1] - t_mdp[0]
    t_spline = t_mdp
    t_spline[1:] += alpha * dt
    return t_spline


def normalise_to_angle(x, m):
    """
    Implements normalisation [-1, 1] --> [0, m * pi]
    """
    return (x + 1) * m * np.pi / 2


def normalise_from_angle(x, m):
    """
    Inverse to normalise_to_angle, [0, m * pi] --> [-1, 1]
    """
    return 2 * x / (m * np.pi) - 1


def ket_to_real(ket):
    """
    Splits real and imag parts of state vector
    """
    a = ket[0].real
    b = ket[0].imag
    c = ket[1].real
    d = ket[1].imag
    return a, b, c, d


def real_to_ket(a, b, c, d):
    """
    Creates numpy array state vector from real values
    """
    return np.array([a + 1j * b, c + 1j * d])
