import sys
import gym
import copy
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import joblib
from tqdm import tqdm
from collections import deque
from torch.utils.data import DataLoader
from utils.buffers import ReplayBuffer
from utils.data import convert_to_numpy, convert_to_tensors, buffer_to
from torch.utils.tensorboard import SummaryWriter

class Algorithm:
    def __init__(self, model_name, device, save_dir: str, env, checkpoint_freq: int, eval_freq: int):
        self.env = env
        # saving info
        self.save_dir = save_dir
        self.model_name = self.name + '_' + model_name
        self.save_file = save_dir + '/' + self.model_name
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.device = device

        # tracking training progress
        self.writer = SummaryWriter(log_dir=self.save_dir+'/logs/'+model_name)
    
    @property
    def name(self):
        return self.__class__.__name__

    def train(self):
        raise NotImplementedError
    
    def evaluate(self, steps_at_eval, max_steps=None):
        raise NotImplementedError


class OffPolicyAlgorithm(Algorithm):
    def __init__(self,
                model_name, 
                device,
                ERBuffer: ReplayBuffer,
                ER_batch_sz: int,
                save_dir: str, 
                env, 
                max_env_steps: int=-1,
                n_rollouts: int=1,
                update_freq: int=4,
                checkpoint_freq: int=-1, 
                eval_freq: int=5000,
            ):
        super().__init__(model_name, device, save_dir, env, checkpoint_freq, eval_freq)
        self.ERBuffer = ERBuffer
        self.ERbatch_sz = ER_batch_sz

        # learning parameters
        self.n_rollouts = n_rollouts
        self.update_freq = update_freq

        # env parameters
        self.max_env_steps = max_env_steps
        self.env = env

        # trackers
        self.total_steps = None
        self.progress = {'evaluation': {'mean_reward': [], 'std_reward': [], 'steps': []}}

        
    def train(self, max_steps, start_steps):
        self.save_models()

        # initialize algorithm
        self.setup_algo()
        self.train_mode()
        self.init_buffer(start_steps)

        checkpoint_steps = 0
        eval_steps = 0
        self.total_steps = 0
        avg_reward = deque([], maxlen=100)

        print('\nRunning algorithm {} on device {}\n'.format(self.name, self.device))
        
        while self.total_steps <= max_steps:
            rollout_info = self.rollouts()
            
            # self.total_steps += rollout_info['steps']
            checkpoint_steps += rollout_info['steps']
            eval_steps += rollout_info['steps']
            avg_reward.append(rollout_info['mean_reward'])

            # gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout_info['steps']
            # self.update(gradient_steps)
            
            if checkpoint_steps >= self.checkpoint_freq and self.checkpoint_freq > 0:
                # print('\nCheckpointing model at {} steps...'.format(total_steps+1))
                save_file = self.save_file+'_steps_{}'.format(self.total_steps)
                self.save_results(save_file)
                checkpoint_steps %= self.checkpoint_freq

            if eval_steps >= self.eval_freq and self.eval_freq > 0:
                self.evaluate()
                eval_steps %= self.eval_freq

            win_avg_reward = sum(avg_reward)/len(avg_reward)
            self.print_progress(rollout_info, win_avg_reward)
            self.writer.add_scalars('Learning Progress', 
                            {'reward': rollout_info['mean_reward'], 
                            'avg_reward': win_avg_reward}, self.total_steps)

        self.save_results()
        with open(self.save_file+'_history.pkl', 'wb') as f:
            joblib.dump(self.progress, f)

    def rollouts(self):
        rollout_reward = 0
        rollout_steps = 0
        rollout_episodes = 0

        for _ in range(self.n_rollouts):
            ep_steps = 0
            ep_reward = 0
            state = self.env.reset()
            while True:
                inputs = convert_to_tensors(state, device=self.device)
                
                action = self.policy(inputs)
                action = convert_to_numpy(action[0])
                
                next_state, r, done, _ = self.env.step(action)
                
                self.ERBuffer.append(state, action, r, next_state, done)

                state = next_state
                ep_steps += 1
                ep_reward += r

                # update total steps
                self.total_steps += 1
                if self.total_steps % self.update_freq == 0:
                    self.update(1)

                if done:
                    break
                if self.max_env_steps > 0 and ep_steps > self.max_env_steps:
                    break
            
            rollout_episodes += 1
            rollout_steps += ep_steps
            rollout_reward += ep_reward
        
        return {'mean_reward': rollout_reward/rollout_episodes, 'n_episodes': rollout_episodes, 'steps': rollout_steps}
    
    def update(self, gradient_steps):
        raise NotImplementedError

    def init_buffer(self, start_steps=None):
        print("Filling buffer with initial random steps...")

        start_steps = self.ERbatch_sz if (start_steps is None or start_steps < self.ERbatch_sz) else start_steps

        state = self.env.reset()
        pbar = tqdm(total=start_steps)
        while len(self.ERBuffer) < start_steps:
            inputs = convert_to_tensors(state, device=self.device)
            
            action = self.policy(inputs)
            action = convert_to_numpy(action[0])

            new_state, reward, done, _ = self.env.step(action)
            self.ERBuffer.append(state, action, reward, new_state, done)

            state = new_state
            pbar.update(1)

            if done:
                state = self.env.reset()

    def policy(self, inputs):
        raise NotImplementedError
    
    def save_models(self):
        raise NotImplementedError

    def save_results(self, save_file=None):
        raise NotImplementedError

    def setup_algo(self):
        raise NotImplementedError
    
    def eval_mode(self):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError

    def evaluate(self):
        self.eval_mode()
        trials = 5
        reward = []
        # steps = []
        for _ in range(trials):
            ep_reward = 0
            ep_steps = 0
            state = self.env.reset()
            while True:
                inputs = convert_to_tensors(state, device=self.device)
                action = self.policy(inputs)
                action = convert_to_numpy(action[0])                
                next_state, r, done, _ = self.env.step(action)

                state = next_state
                ep_steps += 1
                ep_reward += r

                # check for termination
                if done:
                    break
                if self.max_env_steps > 0 and ep_steps > self.max_env_steps:
                    break

            reward.append(ep_reward)
            # steps.append(ep_steps)

        reward = np.array(reward)
        r_mean, r_std = reward.mean(), reward.std()

        # log statistics
        self.writer.add_scalar('Progress', r_mean, self.total_steps)
        self.progress['evaluation']['mean_reward'].append(r_mean)
        self.progress['evaluation']['std_reward'].append(r_std)
        self.progress['evaluation']['steps'].append(self.total_steps)

        self.train_mode()

    def print_progress(self, rollout_info, average_reward):
        msg = "\r--- Reward: {:.2f} -- Steps: {} -- Avg. Reward: {:.2f} -- Total Steps: {} ---".format(
                                rollout_info['mean_reward'], rollout_info['steps'], average_reward, self.total_steps
                            )
        sys.stdout.write(msg)
        sys.stdout.flush()