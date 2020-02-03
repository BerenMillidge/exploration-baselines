from baselines.envs import TorchEnv, NoisyEnv, const
import torch
import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODE_LEN = 500

def plot_trajectory(env,trajectory):
    traj = np.array(trajectory)
    xs = traj[0,:]
    ys = traj[1,:]
    xdots = traj[2,:]
    ydots = traj[3,:]
    fig, axs = plt.subplots(nrows=2,ncols=1)
    ax1, ax2 = axs
    ax1.set_title("x-y position")
    ax1.set_xlim(-env._env.xwidth, env._env.xwidth)
    ax1.set_ylim(-env._env.ywidth, env._env.ywidth)
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.scatter(xs, ys)

    ax2.set_title("x-y velocity")
    ax2.set_xlim(-env._env.velocity_lim, env._env.velocity_lim)
    ax2.set_ylim(-env._env.velocity_lim, env._env.velocity_lim)
    ax2.set_xlabel("X velocity")
    ax2.set_ylabel("Y velocity")
    ax2.scatter(xdots, ydots)

    plt.show()



if __name__ == "__main__":
    env = TorchEnv(const.GRID_WORLD_SEARCH, MAX_EPISODE_LEN)
    s = env.reset()
    trajectories = []

    for _ in range(MAX_EPISODE_LEN):
        a = env.sample_action()
        s, r, d = env.step(a)
        print(r)
        #env.render()
        trajectories.append(s)
        if d:
            break

    plot_trajectory(env, trajectories)
    env.close()
