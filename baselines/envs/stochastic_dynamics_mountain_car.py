import math
import gym
import numpy as np
from baselines.envs import SparseMountainCar

class StochasticMountainCar(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self,noise_func="heteroscedastic",uniform_std=0.001,
    heteroscedastic_std=0.005,
    heteroscedastic_min_std=0.001,
    max_threshold=0,
    min_threshold = -1.2,
     goal_velocity=0,
     no_penalty=True):
        self.noise_func = noise_func
        self.goal_velocity = goal_velocity
        self.no_penalty = no_penalty
        self.uniform_std = uniform_std
        self.heteroscedastic_std = heteroscedastic_std
        self.heteroscedastic_min_std = heteroscedastic_min_std
        if self.noise_func == "uniform":
            self.noise_func = self.uniform_noise

        if self.noise_func == "heteroscedastic":
            self.noise_func = self.heteroscedastic_position_noise

        self.env = SparseMountainCar()
        self.env.goal_velocity = self.goal_velocity
        self.env.no_penalty = self.no_penalty
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

    def seed(self, seed=None):
        return self.env.seed(seed)

    def set_noise(self, noise_func):
        if noise_func == "uniform":
            self.noise_func = self.uniform_noise

        if noise_func == "heteroscedastic":
            self.noise_func = self.heteroscedastic_position_noise



    def reset(self):
        return self.env.reset()

    def _height(self,xs):
        return self.env.height(xs)

    def render(self,mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state += self.noise_func(state)
        #check noise doesn't overstep bounds
        state[0] = min(max(state[0],self.env.min_position), self.env.max_position)
        state[1] = max(min(state[1], self.env.max_speed),-self.env.max_speed)
        self.env.state = state
        bool(state[0] >= self.env.goal_position and state[1] >= self.env.goal_velocity)
        return state, reward, done, info

    def uniform_noise(self,state):
        return np.random.normal(0, self.uniform_std, size=state.shape)

    def heteroscedastic_position_noise(self, state):
        diff = self.env.max_position - state[0]
        #print(diff)
        n = np.random.normal(0, max(diff * self.heteroscedastic_std, self.heteroscedastic_min_std), size=state.shape)
        #print(n)
        return n

    def threshold_noise(self,state):
        if state[0] <= self.max_threshold and state[0] >= self.min_threshold:
            n = np.random.normal(0,self.uniform_std,size=state.shape)
            return n
        return 0
