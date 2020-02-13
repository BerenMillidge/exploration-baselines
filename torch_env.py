# pylint: disable=not-callable
# pylint: disable=no-member

import gym
import torch

try:
    import roboschool
    from baselines.envs import (
        SparseMountainCar,
        SparseCartpoleSwingup,
        SparseDoublePendulum,
        SparseHalfCheetah,
        SparseBipedalWalker,
        StochasticMountainCar,
        SparseLunarLander,
        SparseLunarLanderContinuous,
        GridWorldSearch,
        RewardGradientGridWorld,
        BipedalWalker,
        AcrobotEnv,
        LunarLander,
        LunarLanderContinuous,
        const,
    )
except:
    from baselines.envs import (
        SparseMountainCar,
        SparseDoublePendulum,
        SparseBipedalWalker,
        StochasticMountainCar,
        SparseLunarLander,
        SparseLunarLanderContinuous,
        GridWorldSearch,
        RewardGradientGridWorld,
        BipedalWalker,
        AcrobotEnv,
        LunarLander,
        LunarLanderContinuous,
        const,
    )

try:
    import pybullet
except:
    print("Pybullet not installed. Oh well.")
    pass
try:
    import pybullet_envs
except:
    print("pybullet_envs not installed. Oh well.")
    pass
class TorchEnv(object):
    def __init__(
        self, env_name, max_episode_len, action_repeat=1, device="cpu",return_torch=False, seed=None
    ):

        print(env_name)
        print(const.SPARSE_CARTPOLE_SWINGUP)
        print(env_name == const.SPARSE_CARTPOLE_SWINGUP)

        if env_name == const.SPARSE_MOUNTAIN_CAR:
            self._env = SparseMountainCar()
        elif env_name == const.SPARSE_CARTPOLE_SWINGUP:
            self._env = SparseCartpoleSwingup()
        elif env_name == const.SPARSE_DOUBLE_PENDULUM:
            self._env = SparseDoublePendulum()
        elif env_name == const.SPARSE_HALF_CHEETAH:
            self._env = SparseHalfCheetah()
        elif env_name == const.SPARSE_BIPEDAL_WALKER:
            self._env = SparseBipedalWalker()
        elif env_name == const.STOCHASTIC_MOUNTAIN_CAR:
            self._env = StochasticMountainCar()
        elif env_name == const.SPARSE_LUNAR_LANDER:
            self._env = SparseLunarLander()
        elif env_name == const.SPARSE_LUNAR_LANDER_CONTINUOUS:
            self._env = SparseLunarLanderContinuous()
        elif env_name == const.GRID_WORLD_SEARCH:
            self._env = GridWorldSearch()
        elif env_name == const.REWARD_GRADIENT_GRIDWORLD:
            self._env = RewardGradientGridWorld()
        elif env_name == const.ACROBOT:
            self._env = AcrobotEnv()
        elif env_name == const.BIPEDALWALKER:
            self._env = BipedalWalker()
        elif env_name == const.LUNARLANDER:
            self._env = LunarLander()
        elif env_name == const.LUNARLANDERCONTINUOUS:
            self._env = LunarLanderContinuous()
        else:
            self._env = gym.make(env_name)

        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.done = False
        self.device = device
        self.return_torch = return_torch
        # get maximum and minimum reward estimates for the environment, if they exist
        try:
            self.max_reward = self._env.max_reward
        except:
            self.max_reward = None
        try:
            self.min_reward = self._env.max_reward
        except:
            self.min_reward = None
        if seed is not None:
            self._env.seed(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        state = self._env.reset()
        self.done = False
        if self.return_torch:
            return torch.tensor(state, dtype=torch.float32).to(self.device)
        else:
            return state

    def step(self, action):
        if self.return_torch:
            action = action.cpu().detach().numpy()
        reward = 0

        for _ in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len
            if done:
                self.done = True
                break

        if self.return_torch:
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

        return state, reward, done

    def sample_action(self):
        if self.return_torch:
            return torch.from_numpy(self._env.action_space.sample()).to(self.device)
        else:
            return self._env.action_space.sample()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def state_dims(self):
        return self._env.observation_space.shape

    @property
    def action_dims(self):
        return self._env.action_space.shape

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space