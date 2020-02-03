import math
import gym
import numpy as np

class GridWorldSearch(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, xwidth=50, ywidth=50, velocity_lim=5, random_goal=False, goal_position=[30,-45],action_lim=1,goal_threshold=1,max_len=500):
        self.xwidth =xwidth
        self.ywidth = ywidth
        self.velocity_lim = velocity_lim
        self.random_goal = random_goal
        self.goal_position = goal_position
        self.action_lim = action_lim
        self.goal_threshold = goal_threshold
        self.max_len = max_len
        self.num_steps = 0
        self.dt = 0.1

        self.low_state = np.array([-self.xwidth,-self.ywidth, -self.velocity_lim])
        self.high_state = np.array([self.xwidth,self.ywidth, self.velocity_lim])
        self.action_space = gym.spaces.Box(low=-self.action_lim, high=self.action_lim, shape=(2,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        self.viewer = None
        if self.random_goal:
            self.goal_position = np.random.uniform(low=np.array([-self.xwidth, -self.ywidth]),high=np.array([self.xwidth,self.ywidth]),shape=(2,1))
        self.goalx, self.goaly = self.goal_position

        self.seed()
        self.reset()
        self.state = np.zeros([4,1])
        self.done=False

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.random.normal(0,0.01, size=(4,1))
        self.done = False
        return self.state

    def step(self, action):
        #clip action
        a_x,a_y = action
        a_x = min(max(a_x,-self.action_lim), self.action_lim)
        a_y = min(max(a_y,-self.action_lim), self.action_lim)
        xpos, ypos, velx,vely = self.state
        velx = dt * a_x
        vely = dt * a_y
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
        reward = 0
        if abs(xpos - self.goalx) <=self.goal_threshold and abs(ypos - self.goaly) <= self.goal_threshold:
            reward = 100

        #check if episode has ended
        self.num_steps +=1
        if self.num_steps >= self.max_len:
            self.done=True
            self.num_steps = 0
        self.state = np.array([xpos, ypos, velx, vely])
        return self.state, reward, self.done, {}


    def render(self, mode="human"):
        raise NotImplementedError("Rendering not implemented for this custom environment")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
