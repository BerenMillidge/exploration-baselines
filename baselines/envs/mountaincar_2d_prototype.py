# simplest implementation of a 2d mountain car. Basically it's the same in the second dimension for as long as desired (infinitely)
# or within 2d limits. Just a tiled copy of the 1d version. The reward is only obtained within a small delta of z=0.
#movement along x-y dimension is constrained standardly. Movement in z is unconstrained - simple velocity/acceleration

import math
import gym
import numpy as np
import matplotlib.pyplot as plt

class MountainCar2D(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_velocity=0, no_penalty=True,zmax=10,zmin=-10,zthresh=0.5):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        self.no_penalty = no_penalty
        self.zmax = zmax
        self.zmin = zmin
        self.zthresh = zthresh

        self.low_state = np.array([self.min_position, -self.max_speed,zmin,-1])
        self.high_state = np.array([self.max_position, self.max_speed,zmax,1])

        self.viewer = None

        self.action_space = gym.spaces.Box(
            low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        self.max_reward = 1
        self.min_reward = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        zpos = self.state[2]
        force = min(max(action[0], -1.0), 1.0)
        zvel = min(max(action[0], -1.0), 1.0)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        zpos = min(max(zpos + zvel,self.zmin),self.zmax)
        velocity = min(max(velocity, -self.max_speed),self.max_speed)
        position += velocity
        position = min(max(position, -self.max_position),self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity and self.zpos >=-self.zthresh and self.zpos <= self.zthresh )

        reward = 0
        if done:
            reward = 1.0

        self.state = np.array([position, velocity,zpos,zvel])
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0,0,0])
        return np.array(self.state)

    def state_from_obs(self,obs):
        return obs

    def set_state(self, state):
        self.state = state

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human"):
        raise NotImplementedError("Sorry rendering for 2d env not implemented yet")
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def vel_fn(self, position):
        return  -0.0025 * math.cos(3 * position)

    def plot_velocity_function(self):
        xs = np.linspace(self.min_position,self.max_position, 1000)
        vels = [self.vel_fn(pos) for pos in xs]
        plt.plot(xs, vels)
        plt.show()


if __name__ == '__main__':
    env = MountainCar2D()
    env.plot_velocity_function()
    # so somehow this is the wrong way around, which is really really weird. It would be pretty damn bad if the standard mountain car implementation was incorrec
    s = env.reset()
    print(s)
    a = env.action_space.sample()
    print(a)
    ss = []
    for i in range(100):
      a = env.action_space.sample()
      s,r,done,_ = env.step(a)
      ss.append(s)

    ss = np.array(ss)
    plt.plot(ss[:,0],label="xpos")
    plt.plot(ss[:,1],label="xvel")
    plt.plot(ss[:,2], label="zpos")
    plt.plot(ss[:,3], label="zvel")
    plt.legend()
    plt.show()
    print(ss[:,1])
