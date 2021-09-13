import gym
import pybullet_envs
import torch
from utils.models import  ValidationCriticMLP, ValidationActorMLP
from utils.buffers import ReplayBuffer
from algos.ddpgwd import DDPGwD

def run():
    # env
    env = gym.make('InvertedDoublePendulumBulletEnv-v0')
    env.render()
    env.reset()

    # shape info
    in_feats = env.observation_space.shape[0]
    act_feats = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # actor and critic
    critic = ValidationCriticMLP(in_feats, act_feats, hidden_layers=[400, 300])
    actor = ValidationActorMLP(in_feats, act_feats, hidden_layers=[400,300], action_bound=action_bound)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # replay buffer
    ERBuffer = ReplayBuffer(buffer_size=int(1e6))

    lr = {'actor': 1e-4, 'critic': 1e-3}
    save_dir = 'D:/IIT/mineRL/validateDDPG'
    ER_batch_sz = 64
    tau = 0.001
    gamma = 0.99
    pretrain_file = None
    algo = DDPGwD('run1', device, ERBuffer, ER_batch_sz, actor, critic, save_dir, env, lr, pretrain_file, tau, gamma, clip_gradients=False)
    algo.train(max_steps=int(1e6), start_steps=int(1e3))
    
if __name__ == "__main__":
    run()