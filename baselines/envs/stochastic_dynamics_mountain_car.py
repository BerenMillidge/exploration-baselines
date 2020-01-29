import math
import gym
import numpy as np

class StochasticMountainCar(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self,noise_func="uniform",uniform_std=0.1,heteroscedastic_std=0.1,heteroscedastic_min_std=0.01, goal_velocity=0,no_penalty=True):
        self.noise_func = noise_func
        self.goal_velocity = goal_velocity
        self.no_penalty = no_penalty
        self.uniform_std = uniform_std
        self.heteroscedastic_std = heteroscedastic_std
        self.heteroscedastic_grad = heteroscedastic_min_std
        if self.noise_func == "uniform"
            self.noise_func = self.uniform_noise

        if self.noise_func == "heteroscedastic":
            self.noise_func = self.heteroscedastic_position_noise

        self.env = gym.make("SparseMountainCar")
        self.env.goal_velocity = self.goal_velocity
        self.env.no_penalty = self.no_penalty

    def seed(self, seed=None):
        return self.env.seed(seed)

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
        #ensure you can't go outside range
        state = min(max(state, self.env.min_position),self.env.max_position)
        self.env.state = state
        bool(state[0] >= self.goal_position and state[1] >= self.goal_velocity)
        return state, reward, done, info

    def uniform_noise(self,state):
        return state + np.random.uniform(0, self.uniform_std, size=state.shape)

    def heteroscedastic_position_noise(self, state):
        diff = self.env.max_position - state[0]
        state += np.random.normal(0, max(diff * self.heteroscedastic_std, self.heteroscedastic_min_std), size=state.shape)
        #ensure you can't go outside range
        state = min(max(state, self.env.min_position),self.env.max_position)
        self.env.state = state
        bool(state[0] >= self.goal_position and state[1] >= self.goal_velocity)
        return state, reward, done, info
