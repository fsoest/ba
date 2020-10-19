import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from helpers import rhs
from scipy.interpolate import CubicSpline


PSI_S_0 = np.array([1 + 0j, 0 + 0j])
ALPHA = 0.1

class DST_env(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, t_steps, t_drive, p_drive):
        """
        t_steps: array, times at which MDP is run
        """
        super(DST_env, self).__init__()

        # Create offset MDP time array
        dt = t_steps[1] - t_steps[0]
        self.t_steps = t_steps + ALPHA * dt

        # Reward range
        self.reward_range = (-np.inf, np.inf)

        # Drive qubit functions
        self.t_drive = t_drive
        self.p_drive = p_drive

        # Array for Transducer spline
        self.t_spline = np.zeros(len(self.t_steps) + 1)
        self.t_spline[:-1] = t_steps
        self.t_spline[-1] = self.t_spline[-2] + dt

        # Action space: Theta, Phi of transducer qubit
        self.action_space = spaces.Box(low=np.zeros(2), high=np.full(2, 2 * np.pi), dtype=np.float)

        # self.observation_space = spaces.Tuple(spaces.Box(low=np.zeros(2), high=np.full(2, 2 * np.pi), dtype=np.float32),
        #     spaces.Box(low=np.full(2, -1), high=np.full(2, 1), dtype=np.complex))

        # Observation space: Theta, Phi of drive qubit
        self.observation_space = spaces.Box(low=np.zeros(2), high=np.full(2, 2 * np.pi), dtype=np.float)


    def step(self, action):
        """
        """
        # Create initial values for solver, E = 0, psi_s = psi_s
        y = np.zeros(3)
        y[1:] = self.psi_s

        # Create transducer qubit function with action values
        self.t_trans_array[self.t + 1] = action[0]
        self.p_trans_array[self.t + 1] = action[1]
        t_trans = CubicSpline(self.t_spline, self.t_trans_array)
        p_trans = CubicSpline(self.t_spline, self.p_trans_array)

        # Solve IVP for next span
        span = (self.t_steps[self.t], self.t_steps[self.t + 1])
        args = (self.t_drive, self.d_drive, t_trans, p_trans, H_dst)
        res = solve_ivp(rhs, span, y, args=args)

        reward = res.y[0][-1]
        self.state = res.y.T[-1][1:]

        # Next time step
        self.t += 1

        if self.t == len(self.t_steps):
            done = True
        else:
            done = False

        return self.state, reward, done, {}


    def reset(self):
        """
        """
        self.psi_s = PSI_S_0
        self.t = 0
        # Set state to Phi, Theta of Drive bit to t_mdp = 0
        self.state = np.array([self.t_drive(self.t_steps[self.t]), self.p_drive(self.t_steps[self.t])])

        # Initialize Transducer Splines
        self.t_trans_array = np.zeros(len(self.t_steps) + 1)
        self.p_trans_array = np.zeros(len(self.t_steps) + 1)
