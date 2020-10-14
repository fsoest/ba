import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from helpers import rhs
# %%
# Interaction Hamiltonian
H_i = qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap())
# DST Hamiltonian
H_dst = qt.tensor(H_i, qt.qeye(2)) + qt.tensor(qt.qeye(2), H_i)

# %%
N = 30
t_end = 50
t_drive = np.linspace(0, 2 * np.pi, N)
p_drive = np.pi * np.sin(np.linspace(0, 8 * np.pi, N))

t_arr = np.linspace(0, t_end, N)

# %%
t_transducer = t_drive #np.random.uniform(0, 2 * np.pi, N)
p_transducer = np.pi * np.cos(np.linspace(0, 2 * np.pi, N)) #np.random.uniform(0, 2 * np.pi, N)

# Spline functions
theta_d = CubicSpline(t_arr, t_drive)
phi_d = CubicSpline(t_arr, p_drive)
theta_t = CubicSpline(t_arr, t_transducer)
phi_t = CubicSpline(t_arr, p_transducer)


args = (theta_d, phi_d, theta_t, phi_t, H_dst)

t_span = (0, 15)

# rhs(0.5, np.array([0, 1+0j, 0+0j]), *args)

E = solve_ivp(rhs, t_span, np.array([0, 1+0j, 0+0j]), args=args)

# %%

states = []
b = qt.Bloch()

for i in range(len(E.y[1:].T)):
    b.add_states(qt.Qobj(E.y[1:].T[i]))
b.show()

# %%
plt.plot(E.t, E.y[0].real)
# %%

# Was fordern? Anfang = End?
# Optimierung
 # RL - Annealing
 # Gradient ascent
 # Diskrete phi, theta
 # Wissen über Systemqubit?
 # geschlossenes Quantensystem markovian?
 # Vorgehen über gym mit Referenzalgorithmen
