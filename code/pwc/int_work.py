from scipy.integrate import dblquad
import numpy as np

def alpha_func(theta_d, phi_d):
    return np.sin(theta_d)*np.exp(1j*phi_d)/2 - 1/2

def a_func(alpha, dT, theta_rho):
    x = np.abs(alpha) * dT
    t1 = -1j*alpha / np.abs(alpha) * np.sin(2 * x) * np.cos(theta_rho)
    t2 = alpha / np.conj(alpha) * np.sin(x)**2 * np.sin(theta_rho)
    t3 = np.sin(theta_rho) * np.cos(x)**2
    return t1 + t2 + t3

def w(a):
    return -1/2 * np.real(a) + 1/2 * np.sqrt(np.real(a)**2 + np.imag(a)**2)


def wrapper(theta_d, phi_d, dT=5, theta_rho=0):
    alpha = alpha_func(theta_d, phi_d)
    a = a_func(alpha, dT, theta_rho)
    return w(a) * np.sin(theta_d)

dblquad(wrapper, 0, 2 * np.pi, lambda x: 0, lambda x: np.pi, )[0]/np.pi


import matplotlib.pyplot as plt
T = np.linspace(0, 10, 50)
phi = np.linspace(0, 2*np.pi, 50)
# %%

e = wrapper(np.pi/2, phi, dT=0, theta_rho=0)
plt.plot(phi, e)
