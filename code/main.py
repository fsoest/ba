import numpy as np
from cont_env import DST_env
from helpers import t_spline
import qutip as qt
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import VecCheckNan, DummyVecEnv
from stable_baselines.common.env_checker import check_env

t_start = 0
t_end = 10
N_mdp = 11
N_spline = N_mdp + 1

t_mdp = np.linspace(t_start, t_end, N_mdp)
t_spline = t_spline(t_mdp)

t_drive = np.linspace(0, 2 * np.pi, N_spline)
p_drive = np.pi * np.sin(np.linspace(0, 8 * np.pi, N_spline))

# Interaction Hamiltonian
H_i = qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap())
# DST Hamiltonian
H_dst = qt.tensor(H_i, qt.qeye(2)) + qt.tensor(qt.qeye(2), H_i)


env = DST_env(t_mdp, t_drive, p_drive, H_dst)

# %%
model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("cont_env_v1")

# %%
env.reset()
check_env(env)

# %%
obs = env.reset()
i = 0
# %%
t = np.zeros(N_spline)
p = np.zeros(N_spline)
# %%
action, _states = model.predict(obs)
obs, rewards, dones, info = env.step(action)
t[i] = action[0]
p[i] = action[1]
i += 1

# %%
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
theta = CubicSpline(range(len(t)), t)
phi = CubicSpline(range(len(p)), p)
t_plot = np.linspace(0, 10)
plt.plot(t_plot, theta(t_plot))
plt.plot(t_plot, theta(t_plot))
