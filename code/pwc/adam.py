import numpy as np
from scipy.optimize import approx_fprime


def ADAM(func, theta_0, t_end, alpha=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8, args=()):
    """
    Implementation of ADAM optimiser, see https://arxiv.org/pdf/1412.6980.pdf
    """
    theta = theta_0
    N = len(theta_0)
    m = np.zeros(N)
    v = np.zeros(N)
    for t in range(t_end):
        g = approx_fprime(theta, func, eps, *args)
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * g**2
        m = m / (1 - beta_1**(t + 1))
        v = v / (1 - beta_2**(t + 1))
        theta = theta + alpha * m / (np.sqrt(v) + eps)
    return theta
