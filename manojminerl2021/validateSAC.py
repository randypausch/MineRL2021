import gym
import pybullet_envs
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from utils.models import ValidationSquashedGaussianPolicy, ValidationCriticMLP
from algos.sac import SAC, SquashedGaussianDistribution
from utils.buffers import ReplayBuffer

def run():
    # env
    env = gym.make('HalfCheetahBulletEnv-v0')
    env.render()
    env.reset()

    # shape info
    in_feats = env.observation_space.shape[0]
    act_feats = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # distribution to generate stochastic actions
    action_dist = SquashedGaussianDistribution(act_feats)

    # actor and critic
    critic = ValidationCriticMLP(in_feats, act_feats, hidden_layers=[400, 300])
    actor = ValidationSquashedGaussianPolicy(action_dist, in_feats, act_feats, hidden_layers=[400, 300], action_bound=action_bound)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ERBuffer = ReplayBuffer(buffer_size=int(3e5))

    lr = {'actor':7.3e-4, 'critic': 1e-3}
    gamma = 0.99
    ER_batch_sz = 256
    tau = 0.02
    save_dir = 'D:/IIT/mineRL/validateSAC'
    for i in range(10):
        model_name = 'SACvalidate_{}'.format(i)
        algo = SAC(model_name, device, ERBuffer, ER_batch_sz, actor, critic, save_dir, env, lr, tau, gamma, gradient_steps=100, checkpoint_freq=-1)
        algo.train(int(1e6), int(1e4))
    
if __name__ == '__main__':
    run()