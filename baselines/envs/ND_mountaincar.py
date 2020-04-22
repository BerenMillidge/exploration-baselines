#mountain car with N distractor dimensions

import math
import gym
import numpy as np
from copy import deepcopy

class MountainCarND(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, N=2,goal_velocity=0, no_penalty=True):
        self.N = N
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.distractor_position_limit = 2
        self.distractor_speed_limit=0.1
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        self.no_penalty = no_penalty
        self.max_zvel = 0.07
        self.max_zpos = 10
        self.reward_thresh =0.5

        self.low_state = [self.min_position, -self.max_speed]
        self.high_state =[self.max_position, self.max_speed]
        for i in range(N):
            self.low_state += [-self.distractor_position_limit,-self.distractor_speed_limit]
            self.high_state += [self.distractor_position_limit,self.distractor_speed_limit]
        self.low_state = np.array(self.low_state)
        self.high_state = np.array(self.high_state)

        self.viewer = None

        self.action_space = gym.spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1+self.N,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        self.max_reward = 1
        self.min_reward = 0

        self.seed()
        self.reset()

    def reset_with_N(self, N):
        self.N = N
        self.low_state = [self.min_position, -self.max_speed]
        self.high_state =[self.max_position, self.max_speed]
        for i in range(N):
            self.low_state += [-self.distractor_position_limit,-self.distractor_speed_limit]
            self.high_state += [self.distractor_position_limit,self.distractor_speed_limit]
        self.low_state = np.array(self.low_state)
        self.high_state = np.array(self.high_state)

        self.viewer = None

        self.action_space = gym.spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1+self.N,), dtype=np.float32
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
        force = action[0]
        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        velocity = min(max(velocity, -self.max_speed),self.max_speed)
        position += velocity
        position = min(max(position, self.min_position),self.max_position)
        self.state[0] = deepcopy(position)
        self.state[1] = deepcopy(velocity)
        for i in range(self.N):
            pos = self.state[(i*2) + 2]
            vel = self.state[(i*2) + 3]
            force = action[i + 1]
            vel += force * self.power
            vel = min(max(vel, -self.distractor_speed_limit),self.distractor_speed_limit)
            pos += vel
            pos = min(max(pos, -self.distractor_position_limit),self.distractor_position_limit)
            self.state[(i*2) + 2] = deepcopy(pos)
            self.state[(i*2) + 3] = deepcopy(vel)

        done = (position >= self.goal_position and velocity >= self.goal_velocity)
        print("DONE: ",done)
        for i in range(self.N):
            if not (self.state[(i*2)+2] >= -self.reward_thresh and self.state[(i*2)+2] <=self.reward_thresh):
                done = False

        reward = 0
        if done:
            reward = 1.0

        #self.state = np.array([position, velocity,zpos,zvel])
        return self.state, reward, done, {}

    def reset(self):
        real_state = [np.random.uniform(low=-0.6,high=-0.4),0]
        for i in range(self.N):
            real_state += list([0,0])

        self.state = np.array(real_state)
        return self.state

    def state_from_obs(self,obs):
        return obs

    def set_state(self, state):
        self.state = state

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human"):
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

if __name__=='__main__':
    env = MountainCarND(N=5)
    s = env.reset()
    print(s)
    a = env.action_space.sample()
    snext,r,done,info = env.step(a)
    print(snext)
    print(r)
    print(done)
