from pwc_helpers import wrapper
import numpy as np
from N2_analyt import E
from scipy.optimize import minimize


N = 2
dt = 1
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state

t_d = np.array([0.1, 0.1]) * np.pi
p_d = np.array([0.25, 0.1]) * 2 * np.pi

t_t = np.array([0.1, 0.75]) * np.pi
p_t = np.array([0.8, 0.2]) * 2 * np.pi
trans = np.concatenate((t_t, p_t))
wrapper(trans, t_d, p_d, dt, rho_0, N)
E(trans, t_d[0], p_d[0], dt)

min1 = minimize(wrapper, np.full(4, np.pi/3), args=(t_d, p_d, dt, rho_0, N))
min1.fun



min2 = minimize(E, np.full(4, np.pi/3), args=(t_d[0], p_d[0], dt))
min2.fun

(min1.fun - min2.fun)/min2.fun
