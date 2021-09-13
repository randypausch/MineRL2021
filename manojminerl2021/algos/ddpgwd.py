import copy
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.algorithm import OffPolicyAlgorithm
from utils.models import PolicyNetwork, QNetwork
from utils.buffers import ReplayBuffer, DemonstrationBuffer
from utils.data import convert_to_tensors, convert_to_numpy

class DDPGwD(OffPolicyAlgorithm):
    def __init__(self,
                model_name,
                device,
                ERBuffer: ReplayBuffer,
                ER_batch_sz: int,
                actor: PolicyNetwork,
                critic: QNetwork,
                save_dir: str, 
                env, 
                lr: {'critic': 1e-3, 'actor': 1e-4},
                pretrain_file=None,
                tau=0.001,
                gamma=0.99,
                max_env_steps=-1,
                n_rollouts=1,
                gradient_steps=None,
                checkpoint_freq=10000, 
                eval_freq=5000,
                clip_gradients=True,
            ):        
        super().__init__(model_name, device, ERBuffer, ER_batch_sz, save_dir, env, 
                    max_env_steps, n_rollouts, gradient_steps, checkpoint_freq, eval_freq)
        
        self.optimizer_class = optim.Adam
        
        # to calculate critic loss
        self.TD_error = nn.MSELoss()

        # setup actor and critic and target networks
        self.actor = actor
        self.critic = critic
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        self.models = [self.actor, self.critic, self.target_actor, self.target_critic]
        
        # hyperparameters
        self.gamma = gamma
        self.actor_lr = lr['actor']
        self.critic_lr = lr['critic']
        self.tau = tau
        
        # setup optimizers
        self.actor_opt = self.optimizer_class(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = self.optimizer_class(self.critic.parameters(), lr=self.critic_lr)

        # load pretrained policy
        self.pretrained = False
        if pretrain_file is not None:
            self.actor.load_state_dict(torch.load(pretrain_file))
            self.pretrained = True

        # set gradient clipping
        if clip_gradients:
            for p in self.actor.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
            for p in self.critic.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    def update(self, update_steps):
        for _ in range(update_steps):
            exp_batch = self.ERBuffer.sample(self.ERbatch_sz)
            
            # retrieve data from batch 
            states = convert_to_tensors(exp_batch.states, device=self.device)
            actions = convert_to_tensors(exp_batch.actions, device=self.device)
            next_states = convert_to_tensors(exp_batch.next_states, device=self.device)
            rewards = convert_to_tensors(exp_batch.rewards, device=self.device)
            dones = convert_to_tensors(exp_batch.dones, device=self.device)

            # actor-critic model outputs
            Qvals = self.critic(states, actions)
            policy_actions = self.actor(states)

            # critic targets
            with torch.no_grad():
                target_Qvals = self.target_critic(next_states, self.target_actor(next_states))
                y = rewards + self.gamma*target_Qvals*(1 - dones)
            
            # update critic
            loss = self.TD_error(Qvals, y)
            self.critic_opt.zero_grad()
            loss.backward()
            self.critic_opt.step()

            # update actor
            performance = -self.critic(states, policy_actions).mean()
            self.actor_opt.zero_grad()
            performance.backward()
            self.actor_opt.step()

            # update target networks
            self.update_target_networks()
        
    def update_target_networks(self):
        # critic target network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        # actor target network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_results(self, save_file=None):
        if save_file is None:
            save_file = self.save_file
        torch.save(self.actor.state_dict(), save_file+'_actor.wts')
        torch.save(self.critic.state_dict(), save_file+'_critic.wts')

    def save_models(self, save_file=None):
        if save_file is None:
            save_file = self.save_file
        with open(save_file+'_models.pkl', 'wb') as f:
            joblib.dump([self.actor, self.critic], f)
    
    def setup_algo(self):
        # initialize target networks with weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # transfer all models to device
        for model in self.models:
            model.to(self.device)
    
    def train_mode(self):
        for model in self.models:
            model.train()
    
    def eval_mode(self):
        for model in self.models:
            model.eval()

    def policy(self, inputs):
        with torch.no_grad():
            pi = self.actor(inputs)
        return pi