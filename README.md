# Exploration baselines

This repository contains a number of environments and algorithms for exploration in RL, with a particular focus on model-based RL. 

## Constraints

- Only consider *continuous* actions
- Only open source environments (i.e. not `MuJoCo`)

# Environments

## MountainCar

A continuous-action version of the mountain car problem.


Used in:

- [VIME](https://arxiv.org/abs/1605.09674)
- [MAX](https://arxiv.org/abs/1810.12162)

## CartpoleSwingup

Used in:

- [VIME](https://arxiv.org/abs/1605.09674)

## Sparse Half Cheetah

A reward of +1 is achieved when the cheetah moves over five units

Used in:

- [VIME](https://arxiv.org/abs/1605.09674)


# Future

## Ant Maze

Navigate an ant through a U-shaped maze. Exploration performance is measured as the fraction of states visited. Currently only implemented in `MuJoCo` 

![info][docs/ant_maze.png]

Used in:
- [MAX](https://arxiv.org/abs/1810.12162)