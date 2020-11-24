import numpy as np
from rand_env import DST_env
import qutip as qt
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
import tensorflow as tf

t_start = 0
t_end = 10
N_mdp = 11

t_mdp = np.linspace(t_start, t_end, N_mdp)

# Interaction Hamiltonian
H_i = qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap())
# DST Hamiltonian
H_dst = qt.tensor(H_i, qt.qeye(2)) + qt.tensor(qt.qeye(2), H_i)

env = DST_env(t_mdp, H_dst)

# %%
model = SAC.load('random_3', env=env)
# %%
env.evaluate(model)
