import numpy as np
from scipy.optimize import minimize


def E(trans, t_d, p_d, dt):
    #t_t, t_prime, p_t, p_prime = trans
    t_t = trans[:, 0]
    t_prime = trans[:, 1]
    p_t = trans[:, 2]
    p_prime = trans[:, 3]
    alpha = np.sqrt(np.sin(t_d)**2 + np.sin(t_t)**2 + 2 * np.sin(t_t) * np.sin(t_d) * np.cos(p_t - p_d))/2
    scale = np.sin(2 * alpha * dt) / alpha / 4
    b = np.sin(t_t) * np.cos(p_t) + np.sin(t_d) * np.cos(p_d)
    c = np.sin(t_prime) * np.sin(p_prime) - np.sin(t_t) * np.sin(p_t)
    d = np.sin(t_t) * np.sin(p_t) + np.sin(t_d) * np.sin(p_d)
    e = np.sin(t_prime) * np.cos(p_prime) - np.sin(t_t) * np.cos(p_t)
    return -scale * (b * c - d * e)


def inter(theta, phi):
    return np.sin(theta) * np.exp(1j * phi) / 2


def E_eigen(trans, drive, dt, rho_0):
    t_t, t_prime, p_t, p_prime = trans
    t_d, x, p_d, y = drive
    alpha = inter(t_d, p_d) + inter(t_t, p_t)
    dtau = inter(t_prime, p_prime) - inter(t_t, p_t)
    return np.real(dtau * rho_0[0, 1]), np.real(dtau * rho_0[1, 0] * np.conj(alpha) / alpha)
    return 2 * (np.cos(np.abs(alpha) * dt)**2 *  + np.sin(np.abs(alpha) * dt)**2 * np.real(dtau * rho_0[1, 0] * np.conj(alpha) / alpha))
