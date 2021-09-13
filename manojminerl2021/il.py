import minerl
import gym
import torch
from utils.models import SquashedGaussianPolicy, QNetwork
from algos.sac import SAC
from utils.distributions import SquashedGaussianDistribution
from utils.buffers import ReplayBuffer, DemonstrationBuffer
from utils.env import MineRLEnv

def run():
    save_dir = '/media/user/997211ec-8c91-4258-b58e-f144225899f4/MinerlV2/dhruvlaad/data/sqil_results'

    # env
    env = gym.make('MineRLTreechopVectorObf-v0')
    # shape info
    action_shape = env.action_space['vector'].shape
    action_bound = env.action_space['vector'].high[0]
    env = MineRLEnv(env)

    # distribution to generate stochastic actions
    action_dist = SquashedGaussianDistribution(action_shape[0])

    # actor and critic
    critic = QNetwork(channels=[32,64,64], fc_layers=[1024,512,512], in_channels=4)
    actor = SquashedGaussianPolicy(action_distribution=action_dist,
                            action_bound=action_bound, channels=[32,64,64], fc_layers=[1024,512])
    
    # replay buffers
    ERBuffer = ReplayBuffer(buffer_size=int(1e6))
    demonstration_envs = ['MineRLObtainDiamondVectorObf-v0', 'MineRLObtainIronPickaxeVectorObf-v0', 'MineRLTreechopVectorObf-v0']
    # save demo buffer
    ddbuffer_file = save_dir + '/ddbuffer.sav'
    DDBuffer = DemonstrationBuffer(envs=demonstration_envs, trim=True, trim_reward=[11,11,64], load_file=ddbuffer_file)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # SQIL + SAC
    algo_params = {'model_name': 'sqil_demo_OIP_OD_eval_TC',
            'device': device, 'ERBuffer': ERBuffer, 
            'ER_batch_sz': 256,
            'policy': actor, 'qfn': critic, 'save_dir': save_dir,
            'env': env,
            'lr': {'actor': 3e-4, 'critic': 1e-3},
            'tau': 0.02, 'gamma': 0.99,
            'max_env_steps': 3000,
            'update_freq': 4,
            'eval_freq': int(1e5),
            'use_sqil': True,
            'DDBuffer': DDBuffer}
    algo = SAC(**algo_params)
    algo.train(max_steps=int(1e6), start_steps=int(1e4))

if __name__ == "__main__":
    run()