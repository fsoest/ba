import numpy as np

def spherical(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def ket(theta, phi):
    return np.array([np.cos(theta/2), np.sin(theta/2) * np.exp(1j * phi)])

t_prime = 1.7 * np.pi
theta = 2 * np.pi - t_prime
phi = 1.4 * np.pi

spherical(theta, phi)
spherical(t_prime, phi + np.pi)

np.cos(theta)
np.cos(t_prime)
