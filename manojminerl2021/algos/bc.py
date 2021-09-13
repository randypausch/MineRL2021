from utils.models import PolicyNetwork
from torch.utils.data import DataLoader
from utils.data import buffer_to, convert_to_numpy, convert_to_tensors
from torch.utils.tensorboard import SummaryWriter
import gym
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

class BehaviourCloning:
    def __init__(self, 
            model: PolicyNetwork, 
            lr: float, 
            trainloader: DataLoader, 
            valloader: DataLoader, 
            save_dir: str, 
            env: gym.Env, 
            eval_freq: int, 
            checkpoint_freq: int,
            device='cpu'
        ):
        self.name = self.__class__.__name__
        self.env = env
        self.model = model

        # saving info
        self.save_dir = save_dir
        self.model_name = self.name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_file = save_dir + '/' + self.model_name
        
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        
        self.device = device

        # parameters for the algorithm
        self.lr = lr
        self.trainloader = trainloader
        self.valloader = valloader
        self.batch_size = trainloader.batch_size
        self.env = env
        self.eval_freq = eval_freq
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # tracking progress
        self.running_loss = {'train': 0.0, 'val': 0.0}
        self.epoch_loss = {'train': [], 'val': []}
        self.writer = SummaryWriter(log_dir=self.save_dir+'/logs_'+datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    def train(self, epochs):
        print('Running algorithm {} on device {}'.format(self.name, self.device))
        # pickle model
        self.save_model()
        # initialize algorithm
        self.model.to(self.device)

        for epoch in range(epochs):
            self.model.train()
            print("Epoch: [{}/{}]".format(epoch+1, epochs))
		
            # reset trackers
            self.running_loss = {'train': [0.0, 0], 'val': [0.0, 0]}
            with tqdm(total=len(self.trainloader.dataset)) as pbar:
                for sample in self.trainloader:
                    inputs, actions = sample
                    
                    # povs, obfvecs = inputs['pov'].to(self.device), inputs['obfvec'].to(self.device)
                    # actions = actions['action'].to(self.device)
                    inputs = buffer_to(inputs, self.device)
                    actions = buffer_to(actions, self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, actions['action'])
                    loss.backward()
                    self.optimizer.step()
                
                    self.running_loss['train'][0] += loss.item()
                    self.running_loss['train'][1] += 1

                    pbar.update(self.batch_size)

            # carry out validation
            self.model.eval()
            with torch.no_grad():
                for data in self.valloader:
                    inputs, actions = data
                    
                    # povs, obfvecs = inputs['pov'].to(self.device), inputs['obfvec'].to(self.device)
                    # actions = actions['action'].to(self.device)
                    inputs = buffer_to(inputs, self.device)
                    actions = buffer_to(actions, self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, actions['action'])

                    self.running_loss['val'][0] += loss.item()
                    self.running_loss['val'][1] += 1


                self.print_info(epoch)
                
                if (epoch + 1) % self.eval_freq == 0:
                    self.evaluate(steps_at_eval=epoch+1)

            if (epoch + 1) % self.checkpoint_freq == 0:
                self.save_results(save_file=self.save_file+'_epoch{}'.format(epoch))

        print("Finished training, saving model to {}".format(self.save_file))
        self.save_results()

    def print_info(self, epoch):
        train_loss = self.running_loss['train'][0]/self.running_loss['train'][1]
        val_loss = self.running_loss['val'][0]/self.running_loss['val'][1]
        print('Epoch: {} - Train Loss: {} - Val Loss: {}'.format(epoch+1, train_loss, val_loss))

    def save_model(self, save_file=None):
        if save_file is None:
            save_file = self.save_file
        with open(save_file+'_model.pkl', 'wb') as f:
            joblib.dump(self.model, f)

    def evaluate(self, steps_at_eval, max_steps=None):
        self.model.eval()
        trials = 10
        reward = []
        # steps = []
        for _ in range(trials):
            ep_reward = 0
            ep_steps = 0
            state = self.env.reset()
            while True:
                inputs = convert_to_tensors(state, device=self.device)
                action = self.model(inputs)
                action = convert_to_numpy(action[0])                
                next_state, r, done, _ = self.env.step(action)

                state = next_state
                ep_steps += 1
                ep_reward += r

                # check for termination
                if done:
                    break
                if max_steps is not None:
                    if ep_steps > max_steps:
                        break

            reward.append(ep_reward)
            # steps.append(ep_steps)

        reward = np.array(reward)
        r_mean, r_std = reward.mean(), reward.std()
        self.writer.add_scalars('Progress', {'reward': r_mean, 'ub': r_mean + r_std/4, 'lb': r_mean - r_std/4}, steps_at_eval)
        
        self.model.train()
    
    def save_results(self, save_file=None):
        if save_file is None:
            save_file = self.save_file
        torch.save(self.model.state_dict(), save_file)