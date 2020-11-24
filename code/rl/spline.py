from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

t_arr = np.linspace(0, 1, 3)
theta = np.array([0, 2 * np.pi, 0])

t = np.linspace(0, 1, 100)
theta_func = CubicSpline(t_arr, theta)

plt.plot(t, theta_func(t))
plt.plot(t, theta_func.derivative()(t))
