import numpy as np
from rand_env import DST_env
import qutip as qt
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
import tensorflow as tf
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

t_start = 0
t_end = 10
N_mdp = 30

t_mdp = np.linspace(t_start, t_end, N_mdp)

# Interaction Hamiltonian
H_i = qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap())
# DST Hamiltonian
H_dst = qt.tensor(H_i, qt.qeye(2)) + qt.tensor(qt.qeye(2), H_i)

env = DST_env(t_mdp, H_dst)

# %%
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
policy_kwargs = dict(layers=[100, 100, 100, 100])

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, policy_kwargs=policy_kwargs)

# %%
model.learn(total_timesteps=20000)
model.save('td3')
# %%
env.evaluate(model)
