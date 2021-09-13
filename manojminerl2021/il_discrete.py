import joblib
import minerl
import gym
import torch
import logging
from utils.models import SoftQNetwork, DiscreteStochasticPolicy
from algos.discrete_sac import DiscreteSAC
from utils.distributions import Categorical
from utils.discretizing import Discretizer
from utils.buffers import ReplayBuffer, DemonstrationBuffer
from utils.env import MineRLEnv

logging.basicConfig(level=logging.INFO)

def run():
    save_dir = '/media/user/997211ec-8c91-4258-b58e-f144225899f4/MinerlV2/dhruvlaad/data/sqil_results'

    # discretizer to deal with discrete actions
    discretizer = Discretizer()
    # load action clustering data
    discretizer.load_cluster_data(save_dir+'/cluster_data.sav')
    
    # create policy and critic networks
    action_dist = Categorical(discretizer.n_actions)
    critic = SoftQNetwork(n_actions=discretizer.n_actions)
    actor = DiscreteStochasticPolicy(action_dist, n_actions=discretizer.n_actions)

    # buffers
    ERBuffer = ReplayBuffer(buffer_size=int(1e6))
    demonstration_envs = ['MineRLObtainDiamondVectorObf-v0', 'MineRLObtainIronPickaxeVectorObf-v0']
    # save demo buffer
    ddbuffer_file = save_dir + '/ddbuffer.sav'
    DDBuffer = DemonstrationBuffer(envs=demonstration_envs, trim=True, trim_reward=[11,11], load_file=ddbuffer_file)
    DDBuffer.discretize_actions(discretizer)

    # create mineRL env
    env = gym.make('MineRLObtainDiamondVectorObf-v0')
    env = MineRLEnv(env, discretizer)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    algo_params = {'model_name': 'sqil_demo_OIP_OD_eval_TC',
            'device': device, 'ERBuffer': ERBuffer, 
            'ER_batch_sz': 256,
            'policy': actor, 'qfn': critic, 'save_dir': save_dir,
            'env': env,
            'lr': {'actor': 3e-4, 'critic': 1e-3},
            'tau': 0.02, 'gamma': 0.99,
            'max_env_steps': 10000,
            'update_freq': 4,
            'eval_freq': int(5e5),
            'use_sqil': True,
            'DDBuffer': DDBuffer}
    algo = DiscreteSAC(**algo_params)
    algo.train(max_steps=int(1e6), start_steps=int(1e4))   
 
if __name__ == "__main__":
    run()