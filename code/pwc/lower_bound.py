import numpy as np
from scipy.linalg import expm
from multiproc.pwc_helpers import wrapper, get_eigen_rho, H_s, int_operator


def lower_bound(drive, N, dt):
    E = np.zeros(N)
    t_d = drive[:N]
    p_d = drive[N:]

    t_t = np.full(N, np.pi/2)
    p_t = np.zeros(N)
    rho = get_eigen_rho(t_d, p_d)[0]

    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])

    # First switching step
    p_t[0] = p_d[0]
    for i in range(N - 1):
        H = H_s(t_d[i], p_d[i], t_t[i], p_t[i])
        U = expm(-1j * H * dt)
        rho = U @ rho @ np.matrix(U).H
        x = np.real(np.trace(sx @ rho))
        y = np.real(np.trace(sy @ rho))
        p_t[i + 1] = (np.arctan2(y, x) + np.pi) % (2 * np.pi)
        dH = int_operator(t_t[i + 1], p_t[i + 1]) - int_operator(t_t[i], p_t[i])
        E[i] = np.real(np.trace(rho @ dH))
    return E, t_t, p_t
