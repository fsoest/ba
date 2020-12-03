import numpy as np


def rho_to_uvw(rho):
    u = rho[:, 1, 0] + rho[:, 0, 1]
    v = 1j * (rho[:, 0, 1] - rho[:, 1, 0])
    w = rho[:, 0, 0] - rho[:, 1, 1]
    return np.array([u, v, w])


def tp_to_uvw(t, p):
    x = np.sin(t) * np.cos(p)
    y = np.sin(t) * np.sin(p)
    z = np.cos(t)
    return np.array([x, y, z])
