import numpy as np
from scipy.linalg import expm
from multiproc.pwc_helpers import wrapper, get_eigen_rho, H_s, int_operator

def lower_bound(drive, N, dt):
    E = np.zeros(N)
    t_d = drive[:N]
    p_d = drive[N:]

    t_t = np.full(N, np.pi/2)
    p_t = np.zeros(N)
    rho_0 = get_eigen_rho(t_d, p_d)[0]

    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])

    # First switching step
    p_t[0] = p_d[0]
    p_t[1] = (p_d[0] + np.pi) % (2 * np.pi)

    H_1 = H_s(t_d[0], p_d[0], t_t[0], p_t[0])
    U_1 = expm(-1j * dt * H_1)
    rho = U_1 @ rho_0 @ np.matrix(U_1).H
    dH = int_operator(t_t[1], p_t[1]) - int_operator(t_t[0], p_t[0])
    E[0] = np.real(np.trace(rho @ dH))

    # Loop over next qubits
    for i in range(N - 2):
        x = np.real(np.trace(sx @ rho))
        y = np.real(np.trace(sy @ rho))
        p_t[i + 2] = (np.arctan2(y, x) + np.pi) % (2 * np.pi)

        # Calculate next rho
        H = H_s(t_d[i + 1], p_d[i + 1], t_t[i + 1], p_t[i + 1])
        U = expm(-1j * dt * H)
        rho = U @ rho @ np.matrix(U).H
        dH = int_operator(t_t[i + 2], p_t[i + 2]) - int_operator(t_t[i + 1], p_t[i + 1])
        E[i + 1] = np.real(np.trace(rho @ dH))

    return E, t_t, p_t
