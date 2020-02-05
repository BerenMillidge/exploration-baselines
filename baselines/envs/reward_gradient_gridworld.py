import math
import gym
import numpy as np

class RewardGradientGridWorld(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, xwidth=20, ywidth=20, velocity_lim=5, reward_gradient=0.1,action_lim=1,max_len=1000):
        self.xwidth =xwidth
        self.ywidth = ywidth
        # x dimension is pointless. Y dimension is the reward gradient.
        self.velocity_lim = velocity_lim
        self.reward_gradient = reward_gradient
        self.action_lim = action_lim
        self.max_len = max_len
        self.num_steps = 0
        self.dt = 1

        self.low_state = np.array([-self.xwidth,-self.ywidth, -self.velocity_lim,-self.velocity_lim])
        self.high_state = np.array([self.xwidth,self.ywidth, self.velocity_lim,self.velocity_lim])
        self.action_space = gym.spaces.Box(low=-self.action_lim, high=self.action_lim, shape=(2,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        self.viewer = None
        if self.random_goal:
            self.goal_position = np.random.uniform(low=np.array([-self.xwidth, -self.ywidth]),high=np.array([self.xwidth,self.ywidth]),shape=(2,1))
        self.goalx, self.goaly = self.goal_position

        self.seed()
        self.reset()
        self.state = np.zeros([4,])
        self.done=False

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.random.normal(0,0.01, size=(4,))
        self.done = False
        return self.state

    def reward_fun(self,xpos, ypos):
        # simplest possible linear gradient
        return reward_gradient * ypos

    def step(self, action):
        #clip action
        a_x,a_y = action
        a_x = min(max(a_x,-self.action_lim), self.action_lim)
        a_y = min(max(a_y,-self.action_lim), self.action_lim)
        #print("as: ",a_x, a_y)
        xpos, ypos, velx,vely = self.state
        velx = self.dt * a_x
        vely = self.dt * a_y
        #clip velocity
        velx = min(max(velx, -self.velocity_lim), self.velocity_lim)
        vely = min(max(vely, -self.velocity_lim), self.velocity_lim)
        #apply update
        xpos += velx
        ypos += vely
        #clip positions
        xpos = min(max(xpos, -self.xwidth), self.xwidth)
        ypos = min(max(ypos,-self.ywidth), self.ywidth)
        #check for goal
        reward = self.reward_fun(xpos, ypos)

        #check if episode has ended
        self.num_steps +=1
        if self.num_steps >= self.max_len:
            self.done=True
            self.num_steps = 0
        self.state = np.array([xpos, ypos, velx, vely]).reshape((4,))
        return self.state, reward, self.done, {}


    def render(self, mode="human"):
        raise NotImplementedError("Rendering not implemented for this custom environment")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
