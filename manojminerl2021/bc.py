from utils.models import PolicyNetwork
from utils.data import DemonstrationData, train_val_split
from torch.utils.data import DataLoader
from algos.bc import BehaviourCloning
import gym
import minerl
import torch

def run():
    env = gym.make('MineRLObtainDiamondDenseVectorObf-v0')
    # shape info
    pov_shape = env.observation_space['pov'].shape
    vec_shape = env.observation_space['vector'].shape
    act_shape = env.action_space['vector'].shape
    action_bound = env.action_space['vector'].high[0]

    # create dataset
    data_dir = '/media/user/997211ec-8c91-4258-b58e-f144225899f4/MinerlV2/dhruvlaad/data/ObtainDiamond'
    train_traj, val_traj = train_val_split(data_dir, holdout=0.2)

    train_dataset = DemonstrationData(data_dir, train_traj, pov_shape, vec_shape, act_shape)
    val_dataset = DemonstrationData(data_dir, val_traj, pov_shape, vec_shape, act_shape)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10)
    valloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=10)

    # create policy network
    model = PolicyNetwork(channels=[16,32,32],
            obfvec_fc_sz=[128,256], pov_fc_sz=256,
            merged_fc_sz=[512, 256], in_channels=3,
            pov_shape=pov_shape, vec_shape=vec_shape,
            action_shape=act_shape, action_bound=action_bound)

    # choose GPU
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    save_dir = '/media/user/997211ec-8c91-4258-b58e-f144225899f4/MinerlV2/dhruvlaad/data/ObtainDiamond/models/checkpoints'
    algo = BehaviourCloning(model=model, lr=1e-4, trainloader=trainloader,
                            valloader=valloader, save_dir=save_dir, env=env, eval_freq=5,checkpoint_freq=5)
    algo.train(epochs=50)


if __name__ == "__main__":    
    run()
