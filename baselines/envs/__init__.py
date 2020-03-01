from .const import *
try:
  import roboschool
  from .mountain_car import SparseMountainCar
  from .cartpole_swingup import SparseCartpoleSwingup
  from .double_pendulum import SparseDoublePendulum
  from .half_cheetah import SparseHalfCheetah
  from .bipedal_walker import SparseBipedalWalker
  from .stochastic_dynamics_mountain_car import StochasticMountainCar
  from .sparse_lunar_lander import SparseLunarLander, SparseLunarLanderContinuous
  from .goal_directed_gridworld import GridWorldSearch
  from .reward_gradient_gridworld import RewardGradientGridWorld
  from .acrobot import AcrobotEnv
  from .bipedal_walker_reward import BipedalWalker
  from .lunar_lander import LunarLander, LunarLanderContinuous
  from .pendulum import PendulumEnv
  from .mountaincar_2d_prototype import MountainCar2D
except:
  print("Cannot import roboschool: only importing those envs which don't require it")
  from .mountain_car import SparseMountainCar
  from .double_pendulum import SparseDoublePendulum
  from .bipedal_walker import SparseBipedalWalker
  from .stochastic_dynamics_mountain_car import StochasticMountainCar
  from .sparse_lunar_lander import SparseLunarLander, SparseLunarLanderContinuous
  from .goal_directed_gridworld import GridWorldSearch
  from .reward_gradient_gridworld import RewardGradientGridWorld
  from .acrobot import AcrobotEnv
  from .bipedal_walker_reward import BipedalWalker
  from .lunar_lander import LunarLander, LunarLanderContinuous
  from .pendulum import PendulumEnv
  from .mountaincar_2d_prototype import MountainCar2D

from .torch_env import TorchEnv
from .wrappers import Wrapper, NoisyEnv
try:
    import pybullet
except:
    print("pybullet not installed. Oh well.")
    pass
try:
    import pybullet_env
except:
    print("pybullet_envs not installed. Oh well.")
    pass
