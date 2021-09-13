import gym
import numpy as np
from utils.data import obs_transforms

class MineRLEnv(gym.Wrapper):
    def __init__(self, env, discretizer=None):
        super().__init__(env)
        self.env = env
        self.discretizer = discretizer

    def reset(self):
        obs = self.env.reset()
        return obs_transforms(obs)
    
    def step(self, action):
        if self.discretizer is not None:
            # convert discrete action to continuous
            action = self.discretizer.make_continuous(action)
        # convert action to float64
        action = action.astype(np.float64)
        # convert to dict form to pass to mineRL env
        action = {'vector': action}
        # get mineRL env outputs
        obs, r, done, _ = self.env.step(action)
        obs = obs_transforms(obs)

        return obs, r, done, None
