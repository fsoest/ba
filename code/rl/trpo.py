import numpy as np
from cont_env import DST_env
from helpers import t_spline
import qutip as qt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.trpo_mpi import TRPO

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


model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
