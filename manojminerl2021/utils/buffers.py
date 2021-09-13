import random
import torch
import joblib
import numpy as np
import logging
from collections import namedtuple
from utils.data import obs_transforms
from utils.discretizing import Discretizer

class BufferClass:
    def __init__(self, buffer_size, low_memory):
        self.buffer_sz = buffer_size
        self.low_memory = low_memory
        self.buffer = []
        self.pos = 0
        self.experience = namedtuple('Transition', ['states', 'actions', 'rewards', 'next_states', 'dones'])

    def __len__(self):
        return len(self.buffer)
    
    def append(self, *args):
        e = args

        # do not store next state if using `low_memory`
        if self.low_memory:
            e = list(e)
            del e[3]
            e = tuple(e)

        # if replay buffer not full, append new experience
        if len(self.buffer) < self.buffer_sz:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e
        self.pos += 1
        self.pos = int(self.pos % self.buffer_sz)

    def sample(self, batch_size):
        raise NotImplementedError

    def save_buffer(self, file):
        with open(file, 'wb') as f:
            joblib.dump(self.buffer, f)

    def load_buffer(self, file):
        with open(file, 'rb') as f:
            buffer = joblib.load(f)    
        
        return buffer

class ReplayBuffer(BufferClass):
    def __init__(self, buffer_size, low_memory=True, load_file=None):
        super().__init__(buffer_size, low_memory)
        if load_file is not None:
            self.buffer = self.load_buffer(load_file)

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in idx:
            exp = self.buffer[i]
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            if self.low_memory:
                exp_done = exp[3]
                dones.append(exp_done)
                # if sample was terminal, `next_state` will not be used, fill with zeros
                if exp_done:
                    next_states.append(np.zeros_like(exp[0]))
                else:
                    # sample was not terminal, and there is a consecutive sample in the buffer
                    if i < len(self.buffer) - 1:
                        next_exp, _, _, _ = self.buffer[i + 1]
                        next_states.append(next_exp)
                    # sample was not terminal and there is no consecutive sample in buffer
                    else:
                        next_states.append(states.pop())
                        prev_exp, _, _, _ = self.buffer[i - 1]
                        states.append(prev_exp)
            else:
                dones.append(exp[4])
                next_states.append(exp[3])

        states, next_states = np.array(states), np.array(next_states)
        actions = np.array(actions).astype(np.float32)
        if len(actions.shape) == 1:
            actions = actions.reshape(-1,1)
        rewards, dones = np.array(rewards).reshape(-1,1), np.array(dones).reshape(-1,1)

        return self.experience(states, actions, rewards, next_states, dones)


class DemonstrationBuffer(BufferClass):
    def __init__(self, envs: list, trim: bool, trim_reward: list, save_file=None, load_file=None):
        # buffer size
        super().__init__(-1, True)
        delattr(self, 'buffer')
    
        if load_file is None:
            # create demonstration dataset
            n_samples = self.collate_data(envs, trim, trim_reward)
            if save_file is not None:
                self.save_buffer(save_file)
        else:
            self.load_buffer(load_file)
            n_samples = self.rewards.shape[0]
        
        self.buffer_sz = n_samples
    

    def sample(self, batch_size):
        idx = np.random.randint(0, self.buffer_sz, size=batch_size)

        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx].astype(np.int16)

        next_states = []
        for i in idx:
            # if end of trajectory, the next state is never used, so give zero inputs
            if self.dones[i]:
                next_states.append(np.zeros_like(self.states[i]))
            else:
                next_states.append(self.states[i+1])
        
        next_states = np.array(next_states)
            
        return self.experience(states, actions, rewards, next_states, dones)

    def append(self, *args):
        raise TypeError('cannot append samples to demonstration buffer.')

    def collate_data(self, envs, trim: bool, trim_reward: int):
        if not isinstance(envs, list):
            envs = [envs]
        # if an integer is provided as trim_reward, then use this for all envs
        if not isinstance(trim_reward, list):
            trim_reward = [trim_reward for i in range(len(envs))]
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        n_samples = 0

        import minerl
        for n_demo, env in enumerate(envs):
            env_data = minerl.data.make(env)
            trajectories = env_data.get_trajectory_names()
            for traj in trajectories:
                try:
                    traj_reward = 0
                    # sample trajectory only if its un-corrupted
                    for i, sample in enumerate(env_data.load_data(traj, include_metadata=True)):
                        if i == 0:
                            meta_data = sample[5]
                            # if trimming trajectories, skip those that dont meet required reward
                            if trim and meta_data['total_reward'] < trim_reward[n_demo]:
                                break

                        self.states.append(obs_transforms(sample[0]))
                        self.actions.append(sample[1]['vector'])
                        self.rewards.append(sample[2])
                        self.dones.append(sample[4])
                        n_samples += 1

                        traj_reward += sample[2]

                        # if trimming break when required reward is met
                        if trim and traj_reward >= trim_reward[n_demo]:
                            # makr the end of trimmed trajectory
                            self.dones[-1] = True
                            break

                except TypeError:
                    # sometimes trajectory file is corrupted, if so skip it
                    pass
        
        self.states = np.array(self.states)
        self.actions = np.array(self.actions).astype(np.float32)
        self.rewards = np.array(self.rewards).reshape(-1,1)
        self.dones = np.array(self.dones).reshape(-1,1)

        return n_samples
    
    def load_buffer(self, file):
        buffer = super().load_buffer(file)
        self.states, self.actions, self.rewards, self.dones = buffer

        logging.info('loaded demonstration data with {} samples, state dim: {}, action dim: {}, rewards shape: {}, dones shape: {}'.format(*self.states.shape,
                                self.actions.shape[1], self.rewards.shape, self.dones.shape))

    def save_buffer(self, file):
        with open(file, 'wb') as f:
            joblib.dump([self.states, self.actions, self.rewards, self.dones], f)

    def discretize_actions(self, discretizer: Discretizer):
        self.actions = discretizer.labels.reshape(-1,1)
        logging.info('converted to discrete actions with shape {}'.format(self.actions.shape))