import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from helpers import rhs, t_spline, normalise_to_angle, normalise_from_angle, ket_to_real
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


PSI_S_0 = np.array([1 + 0j, 0 + 0j])
ALPHA = 1e-4

class DST_env(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, t_mdp, H_dst):
        """
        t_mdp: array, times at which MDP is run
        """
        super(DST_env, self).__init__()

        self.t_mdp = t_mdp

        self.H_dst = H_dst
        # Reward range
        self.reward_range = (-np.inf, np.inf)

        self.t_spline = t_spline(self.t_mdp, ALPHA)


        # Action space: Theta, Phi of transducer qubit, [-1, 1]!
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float)

        # self.observation_space = spaces.Tuple(spaces.Box(low=np.zeros(2), high=np.full(2, 2 * np.pi), dtype=np.float32),
        #     spaces.Box(low=np.full(2, -1), high=np.full(2, 1), dtype=np.complex))

        # Observation space: Theta, Phi of drive qubit, state vector
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float)
        # TODO: Modify code to account for normalisation


    def step(self, action):
        """
        Implements a single MDP step
        """
        # Create initial values for solver, E = 0, psi_s = psi_s
        y = np.zeros(3, dtype=np.complex)
        y[1:] = self.psi_s

        # Create transducer qubit function with action values, ([-1, 1] -> [0, pi / 2pi])
        self.t_trans_array[self.t + 1] = normalise_to_angle(action[0], 1)
        self.p_trans_array[self.t + 1] = normalise_to_angle(action[1], 2)
        self.t_trans = CubicSpline(self.t_spline, self.t_trans_array)
        self.p_trans = CubicSpline(self.t_spline, self.p_trans_array)

        # Solve IVP for next span
        span = (self.t_mdp[self.t], self.t_mdp[self.t + 1])
        args = (self.t_drive, self.p_drive, self.t_trans, self.p_trans, self.H_dst)
        res = solve_ivp(rhs, span, y, args=args)

        reward = res.y[0][-1].real
        self.psi_s = res.y.T[-1][1:]
        self.state = np.array([normalise_from_angle(self.t_drive(self.t_mdp[self.t]), 1), \
         normalise_from_angle(self.p_drive(self.t_mdp[self.t]), 2), \
         *ket_to_real(self.psi_s)])

        # Next time step
        self.t += 1

        # TODO: Check done condition
        if self.t + 1 == len(self.t_mdp):
            done = True
        else:
            done = False

        return self.state, reward, done, {}


    def reset(self):
        """
        """

        # Initiate random drive functions
        t_drive = np.random.uniform(low =0, high=np.pi, size=(len(self.t_spline),))
        p_drive = np.random.uniform(low =0, high=2*np.pi, size=(len(self.t_spline),))
        self.t_drive = CubicSpline(self.t_spline, t_drive)
        self.p_drive = CubicSpline(self.t_spline, p_drive)

        # Create random initial position of system qubit
        random_init = np.random.uniform(low =-1, high=1, size=(4,))
        self.psi_s = np.array([random_init[0] + 1j * random_init[1], random_init[2] + 1j * random_init[3]])
        self.psi_s /= np.linalg.norm(self.psi_s)
        self.t = 0

        # Set state to Theta, Phi of Drive bit to t_mdp = 0
        self.state = np.array([normalise_from_angle(self.t_drive(self.t_mdp[self.t]), 1), \
         normalise_from_angle(self.p_drive(self.t_mdp[self.t]), 2), \
         *ket_to_real(self.psi_s)])

        # Initialize Transducer Splines
        self.t_trans_array = np.zeros(len(self.t_mdp))
        self.p_trans_array = np.zeros(len(self.t_mdp))

        return self.state


    def evaluate(self, model):
        """
        """
        self.reset()
        self.total_rewards = []
        self.phi = [0]
        self.theta = [0]
        done = False
        while done == False:
            action, _states = model.predict(self.state)
            obs, reward, done, notes = self.step(action)
            self.total_rewards.append(reward)
            self.theta.append(action[0])
            self.phi.append(action[1])
        plt.plot(range(len(self.total_rewards)), np.cumsum(self.total_rewards), label='Rewards')
        plt.plot(range(len(self.theta)), self.theta, label='Theta trans')
        plt.plot(range(len(self.phi)), self.phi, label='Phi trans')
        t = np.linspace(0, 10, 100)
        plt.plot(t, self.t_drive(t), label='Theta drive')
        plt.plot(t, self.p_drive(t), label='Phi drive')
        plt.legend()
        return self.theta, self.phi
