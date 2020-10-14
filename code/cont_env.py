import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from helpers import rhs

PSI_S_0 = np.array([1 + 0j, 0 + 0j])

class DST_env(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, t_steps, t_drive, p_drive):
        """
        t_steps: array, times at which
        """
        super(DST_env, self).__init__()
        self.t_steps = t_steps
        self.dt = self.t_steps[1] - self.t_steps[0]
        self.reward_range = (-np.inf, np.inf)
        self.t_drive = t_drive
        self.p_drive = p_drive

        # Action space: Theta, Phi of transducer qubit
        self.action_space = spaces.Box(low=np.zeros(2), high=np.full(2, 2 * np.pi), dtype=np.float)

        # self.observation_space = spaces.Tuple(spaces.Box(low=np.zeros(2), high=np.full(2, 2 * np.pi), dtype=np.float32),
        #     spaces.Box(low=np.full(2, -1), high=np.full(2, 1), dtype=np.complex))

        # Observation space: Theta, Phi of drive qubit
        self.observation_space = spaces.Box(low=np.zeros(2), high=np.full(2, 2 * np.pi), dtype=np.float)


    def step(self, action):
        """
        """
        y = np.zeros(3)
        y[1:] = self.psi_s
        span = (self.t, self.t + self.dt)

        args = (theta_d, phi_d, theta_t, phi_t, H_dst)
        res = solve_ivp(rhs, span, y, args=args)


    def reset(self):
        """
        """
        self.psi_s = PSI_S_0
        self.t = self.t_steps[0]
        self.state = np.array([self.t_drive(self.t), self.p_drive(self.t)])
