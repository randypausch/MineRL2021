import torch
import numpy as np
from algos.sac import SAC
from utils.buffers import ReplayBuffer, DemonstrationBuffer
from utils.models import DiscreteStochasticPolicy, SoftQNetwork
from utils.data import convert_to_tensors

class DiscreteSAC(SAC):
    def __init__(self,
            model_name,
            device,
            ERBuffer: ReplayBuffer,
            ER_batch_sz: int,
            policy: DiscreteStochasticPolicy,
            qfn: SoftQNetwork,
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
        super().__init__(model_name, device, ERBuffer, ER_batch_sz, 
                    policy, qfn, save_dir, env, lr, tau, gamma, reward_scaling,
                    max_env_steps, n_rollouts, update_freq, checkpoint_freq, eval_freq, 
                    clip_gradients, auto_entropy_tuning, target_entropy, log_statistics, 
                    use_sqil, DDBuffer)
        
    def update(self, gradient_steps):
        if self.log_statistics:
            qfn1_loss_li, qfn2_loss_li, policy_loss_li, alpha_li, alpha_loss_li = [], [], [], [], []
        
        for _ in range(gradient_steps):
            if self.use_sqil:
                demo_batch = self.DDBuffer.sample(self.ERbatch_sz)
            exp_batch = self.ERBuffer.sample(self.ERbatch_sz)

            if self.use_sqil:
                states, actions, next_states, rewards, dones = self.sqil_experience_replay(demo_batch, exp_batch)
            else:
                tensor_exp_batch = convert_to_tensors((exp_batch.states, exp_batch.actions,
                                exp_batch.next_states, exp_batch.rewards, exp_batch.dones), device=self.device)
                states, actions, next_states, rewards, dones = tensor_exp_batch
            
            pi, log_pi = self.actor.action_log_prob(states)

            # alpha loss
            if self.auto_entropy_tuning:
                alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach()
                E_alpha_loss = (pi.detach() * alpha_loss).sum(axis=1)
                E_alpha_loss = E_alpha_loss.mean()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 1

            # policy loss
            with torch.no_grad():
                softQ_vals = torch.min(self.qfn1(states), self.qfn2(states))
            # expectation of KL-divergence for policy
            E_kl_div = (pi * (alpha*log_pi - softQ_vals)).sum(axis=1)
            policy_loss = E_kl_div.mean()

            # soft Q function loss
            softQ1_vals = self.qfn1(states).gather(1, actions.long())
            softQ2_vals = self.qfn2(states).gather(1, actions.long())
            with torch.no_grad():
                next_pi, next_log_pi = self.actor.action_log_prob(next_states)
                # value function Eq.(3) SAC paper
                Vfunc = torch.min(
                    self.target_qfn1(next_states),
                    self.target_qfn2(next_states)
                ) - alpha*next_log_pi
                # expected value function w.r.t. action probabilities
                E_Vfunc = (next_pi * Vfunc).sum(axis=1).reshape(-1,1)
                softQ_target = self.reward_scaling*rewards + (1 - dones)*self.gamma*E_Vfunc
            
            qfn1_loss = self.qf_criterion(softQ1_vals, softQ_target.float())
            qfn2_loss = self.qf_criterion(softQ2_vals, softQ_target.float())

            # update entropy coef
            if self.auto_entropy_tuning:
                self.alpha_opt.zero_grad()
                E_alpha_loss.backward()
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