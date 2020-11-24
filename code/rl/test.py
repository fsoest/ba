import numpy as np
from rand_env import DST_env
from helpers import t_spline
import qutip as qt
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import VecCheckNan, DummyVecEnv
from stable_baselines.common.env_checker import check_env

t_start = 0
t_end = 10
N_mdp = 11

t_mdp = np.linspace(t_start, t_end, N_mdp)
t_spline = t_spline(t_mdp, 0.1)
t_spline
t_drive = np.linspace(0, 2 * np.pi, N_mdp)
p_drive = np.pi * np.sin(np.linspace(0, 8 * np.pi, N_mdp))

# Interaction Hamiltonian
H_i = qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap())
# DST Hamiltonian
H_dst = qt.tensor(H_i, qt.qeye(2)) + qt.tensor(qt.qeye(2), H_i)


env = DST_env(t_mdp, H_dst)

env.reset()
# %%
env.step(np.array([0.5, -0.5]))
env.p_trans_array


a = np.array([1+ 5j, 3 + 4j])
