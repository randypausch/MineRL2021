from utils.algorithm import OffPolicyAlgorithm
from utils.buffers import ReplayBuffer,DemonstrationBuffer
from utils.models import SquashedGaussianPolicy, QNetwork
from utils.data import convert_to_tensors
import copy
import joblib
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class SAC(OffPolicyAlgorithm):
    def __init__(self,
            model_name,
            device,
            ERBuffer: ReplayBuffer,
            ER_batch_sz: int,
            policy: SquashedGaussianPolicy,
            qfn: QNetwork,
            save_dir: str,
            env,
            lr: {'actor': 3e-4, 'critic': 1e-3},
            tau=0.005,
            gamma=0.99,
            reward_scaling=1.0,
            max_env_steps=-1,
            n_rollouts=1,
            update_freq=4,
            checkpoint_freq=100000, 
            eval_freq=5000,
            clip_gradients=False,
            auto_entropy_tuning=True,
            target_entropy=None,
            log_statistics=False,
            use_sqil=False,
            DDBuffer: DemonstrationBuffer=None
        ):
        super().__init__(model_name, device, ERBuffer, ER_batch_sz, save_dir, env, 
                        max_env_steps, n_rollouts, update_freq, checkpoint_freq, eval_freq)

        self.log_statistics = log_statistics

        # optimizer
        self.optimizer_class = optim.Adam
        # criterion
        self.qf_criterion = nn.MSELoss()

        # setup all required models
        self.actor = policy
        self.qfn1 = copy.deepcopy(qfn)
        self.qfn2 = copy.deepcopy(qfn)
        self.target_qfn1 = copy.deepcopy(self.qfn1)
        self.target_qfn2 = copy.deepcopy(self.qfn2)
        self.models = [self.actor, self.qfn1, self.qfn2, self.target_qfn1, self.target_qfn2]

        # hyperparameters
        self.policy_lr = lr['actor']
        self.qfn_lr = lr['critic']
        self.reward_scaling = reward_scaling
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy_tuning = auto_entropy_tuning

        # set up optimizers
        self.policy_optimizer = self.optimizer_class(self.actor.parameters(), lr=self.policy_lr)
        self.qfn1_optimizer = self.optimizer_class(self.qfn1.parameters(), lr=self.qfn_lr)
        self.qfn2_optimizer = self.optimizer_class(self.qfn2.parameters(), lr=self.qfn_lr)


        if self.auto_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(self.actor.action_shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
            self.alpha_opt = self.optimizer_class([self.log_alpha], lr=self.policy_lr)
        
        # set gradient clipping
        if clip_gradients:
            for p in self.actor.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
            for p in self.qfn1.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
            for p in self.qfn2.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

        # for SQIL mode
        self.use_sqil = use_sqil
        if use_sqil:
            if DDBuffer is not None:
                self.DDBuffer = DDBuffer
            else:
                raise ValueError('when using SQIL mode, you need to provide a demonstrations buffer.')
            
            # for SQIL
            self.ERbatch_sz = int(self.ERbatch_sz/2)

    def update(self, gradient_steps):
        
        if self.log_statistics:
            qfn1_loss_li, qfn2_loss_li, policy_loss_li, alpha_li, alpha_loss_li = [], [], [], [], []

        for _ in range(gradient_steps):
            if self.use_sqil:
                demo_batch = self.DDBuffer.sample(self.ERbatch_sz)
            exp_batch = self.ERBuffer.sample(self.ERbatch_sz)

            # retrieve data from batch 
            if self.use_sqil:
                states, actions, next_states, rewards, dones = self.sqil_experience_replay(demo_batch, exp_batch)
            else:
                tensor_exp_batch = convert_to_tensors((exp_batch.states, exp_batch.actions,
                            exp_batch.next_states, exp_batch.rewards, exp_batch.dones), device=self.device)
                states, actions, next_states, rewards, dones = tensor_exp_batch
        
            pi_actions, log_pi = self.actor.action_log_prob(states)
            log_pi = log_pi.unsqueeze(-1)

            # alpha loss
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 1

            # policy loss
            softQ_vals = torch.min(self.qfn1(states, pi_actions), self.qfn2(states, pi_actions))
            policy_loss = (alpha*log_pi - softQ_vals).mean()

            # soft Q function loss
            softQ1_vals = self.qfn1(states, actions)
            softQ2_vals = self.qfn2(states, actions)
            with torch.no_grad():
                next_pi_actions, next_log_pi = self.actor.action_log_prob(next_states)
                next_log_pi = next_log_pi.reshape(-1, 1)
                target_softQ_vals = torch.min(
                            self.target_qfn1(next_states, next_pi_actions),
                            self.target_qfn2(next_states, next_pi_actions),
                        ) - alpha * next_log_pi
                
                softQ_target = self.reward_scaling*rewards + (1 - dones)*self.gamma*target_softQ_vals
                
            qfn1_loss = self.qf_criterion(softQ1_vals, softQ_target.float())
            qfn2_loss = self.qf_criterion(softQ2_vals, softQ_target.float())

            # update entropy coef
            if self.auto_entropy_tuning:
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
            
            # update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # update Q networks
            self.qfn1_optimizer.zero_grad()
            self.qfn2_optimizer.zero_grad()
            qfn1_loss.backward()
            qfn2_loss.backward()
            self.qfn1_optimizer.step()
            self.qfn2_optimizer.step()

            if self.log_statistics:
                qfn1_loss_li.append(qfn1_loss.item())
                qfn2_loss_li.append(qfn2_loss.item())
                policy_loss_li.append(policy_loss.item())
                alpha_li.append(alpha.detach().cpu().numpy())
                alpha_loss_li.append(alpha_loss.item())

        # soft update target networks
        self.update_target_networks()

        if self.log_statistics:
            qfn1_loss_li, qfn2_loss_li = np.array(qfn1_loss_li), np.array(qfn2_loss_li)
            policy_loss_li = np.array(policy_loss_li)
            alpha_li, alpha_loss_li = np.array(alpha_li), np.array(alpha_loss_li)

            # track metrics
            self.writer.add_scalars('Critic Losses', {'qfn1_loss': qfn1_loss_li.mean(), 'qfn2_loss': qfn2_loss_li.mean()}, self.total_steps)
            self.writer.add_scalar('Policy Loss', policy_loss_li.mean(), self.total_steps)
            if self.auto_entropy_tuning:
                self.writer.add_scalars('Entropy Coefficient', {'alpha': alpha_li.mean(), 'alpha_loss': alpha_loss_li.mean()}, self.total_steps)

    def update_target_networks(self):
        for target_param, param in zip(self.target_qfn1.parameters(), self.qfn1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_qfn2.parameters(), self.qfn2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_results(self, save_file=None):
        if save_file is None:
            save_file = self.save_file
        torch.save(self.actor.state_dict(), save_file+'_policy.wts')
        torch.save(self.qfn1.state_dict(), save_file+'_qfn1.wts')
        torch.save(self.qfn2.state_dict(), save_file+'_qfn2.wts')

    def save_models(self, save_file=None):
        if save_file is None:
            save_file = self.save_file
        with open(save_file+'_models.pkl', 'wb') as f:
            joblib.dump([self.actor, self.qfn1, self.qfn2], f)

    def setup_algo(self):
        # initialize target networks with weights
        self.target_qfn1.load_state_dict(self.qfn1.state_dict())
        self.target_qfn2.load_state_dict(self.qfn2.state_dict())
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
        deterministic = not self.actor.training
        with torch.no_grad():
                pi = self.actor(inputs, deterministic)
        return pi

    # for formatting the rewards for SQIL
    def sqil_experience_replay(self, demo_batch, exp_batch):
        if not self.use_sqil:
            return

        states = np.vstack((exp_batch.states, demo_batch.states))
        next_states = np.vstack((exp_batch.next_states, demo_batch.next_states))
        actions = np.vstack((exp_batch.actions, demo_batch.actions)).astype(np.float32)
        dones = np.vstack((exp_batch.dones, demo_batch.dones))
        exp_rewards, demo_rewards = np.zeros(exp_batch.rewards.shape), np.ones(demo_batch.rewards.shape)
        rewards = np.vstack((exp_rewards, demo_rewards))

        return convert_to_tensors((states, actions, next_states, rewards, dones), device=self.device)