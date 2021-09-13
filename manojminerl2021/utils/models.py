import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import namedtuple
from utils.distributions import Categorical

POV_DIM = 64

class IMPALA(nn.Module):
    def __init__(self, in_channels, channels):
        super(IMPALA, self).__init__()
        
        # each ImpalaBlock contains two FixupResidual blocks and additional two FixupResidual blocks
        n_residuals = 2*len(channels) + 2

        blocks = []
        for channel in channels:
            blocks.append(ImpalaBlock(in_channels, channel, n_residuals))
            in_channels = channel
        
        out_channels = channels[-1]

        # additional 2 FixupResidual blocks
        self.resblock1 = FixupResidual(out_channels, n_residuals)
        self.resblock2 = FixupResidual(out_channels, n_residuals)

        self.blocks = nn.Sequential(*blocks) 
        self.flattened_dim = int((POV_DIM/(2**len(channels)))) ** 2 * out_channels
        
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        out = self.blocks(x)
        out = self.activation(out)
        out = out.view(out.shape[0], -1)
        out = self.activation(out)
        return out

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_residuals):
        super(ImpalaBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock1_ = FixupResidual(out_channels, n_residuals)
        self.resblock2_ = FixupResidual(out_channels, n_residuals)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.resblock1_(out)
        out = self.resblock2_(out)
        return out

class FixupResidual(nn.Module):
    def __init__(self, channels, n_residuals):
        super(FixupResidual, self).__init__()

        self.bias1 = nn.Parameter(torch.zeros([channels, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([channels, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([channels, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([channels, 1, 1]))

        self.scale = nn.Parameter(torch.ones([channels, 1, 1]))

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        # Fixup Initialization
        for p in self.conv1.parameters():
            p.data.mul_(1/np.sqrt(n_residuals))
        for p in self.conv2.parameters():
            p.data.zero_()

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.activation(x)
        out += self.bias1
        out = self.conv1(out)
        out += self.bias2
        out = self.activation(out)
        out += self.bias3
        out = self.conv2(out)
        out *= self.scale
        out += self.bias4
        return out + x 

class MLP(nn.Module):
    def __init__(self, 
            in_feats: int, 
            out_feats: int, 
            hidden_layers: list=[], 
            hidden_activation=nn.ReLU,
            output_activation=nn.ReLU
        ):
        super(MLP, self).__init__()
        
        feats = in_feats
        if len(hidden_layers) > 0:
            fcs = []
            for sz in hidden_layers:
                fcs.extend([nn.Linear(feats, sz), hidden_activation()])
                feats = sz
        
            self.fcs = nn.Sequential(*fcs)
        
        self.output = nn.Sequential(nn.Linear(feats, out_feats), output_activation())
    
    def forward(self, x):
        if hasattr(self, 'fcs'):
            x = self.fcs(x)
        x = self.output(x)
        return x
        
########################################################################################################
#                                              Network models                                          #
########################################################################################################
class QNetwork(nn.Module):
    def __init__(self, 
                channels=[16, 32, 32],
                fc_layers=[1024,512,512],
                in_channels=4,
                input_shape=(64,64,4),
                action_shape=(64,)
            ):
        super(QNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_shape = action_shape

        # feature extractor
        self.feats_extractor = IMPALA(in_channels, channels)

        # fc layers
        fc_in_feats = self.feats_extractor.flattened_dim + self.action_shape[0]
        fc_out_feats = fc_layers[-1]
        del fc_layers[-1]

        self.fc = MLP(fc_in_feats, fc_out_feats, fc_layers)

        # output 
        self.output = nn.Linear(fc_out_feats, 1)

    def forward(self, state, action):
        state = shape_input_tensor(state, self.input_shape)
        action = shape_input_tensor(action, self.action_shape)

        feats = self.feats_extractor(state)

        # merge two branches
        out = torch.cat((feats, action), 1)
        out = self.fc(out)

        out = self.output(out)
        return out

class PolicyNetwork(nn.Module):
    def __init__(self, 
                channels=[16, 32, 32],
                fc_layers=[1024,512],
                in_channels=4,
                input_shape=(64,64,4),
                action_shape=(64,),
                action_bound=1.05
            ):
        super(PolicyNetwork, self).__init__()

        self.action_bound = abs(action_bound)
        self.input_shape = input_shape
        self.action_shape = action_shape
        
        # feature extractor
        self.feats_extractor = IMPALA(in_channels, channels)

        # fc layers
        fc_in_feats = self.feats_extractor.flattened_dim
        fc_out_feats = fc_layers[-1]
        del fc_layers[-1]
        self.fc = MLP(fc_in_feats, fc_out_feats, fc_layers)

        # output
        self.output = nn.Linear(fc_out_feats, self.action_shape[0])

    def forward(self, state):
        state = shape_input_tensor(state, self.input_shape)
        out = self.feats_extractor(state)
        out = self.fc(out)
        out = self.output(out)
        out = self.action_bound*torch.tanh(out)
        return out

########################################################################################################
#                                             misc  models                                             #
########################################################################################################
class ValidationActorMLP(MLP):
    def __init__(self, in_feats: int, action_feats: int, hidden_layers: list, 
            hidden_activation=nn.ReLU, output_activation=nn.Tanh, action_bound=1.05):
        super().__init__(in_feats, action_feats, hidden_layers, hidden_activation, output_activation)
        self.action_bound = abs(action_bound)
        self.state_shape = (in_feats,)
        self.action_shape = (action_feats,)
    
    def forward(self, x):
        x = shape_input_tensor(x.float(), self.state_shape)
        x = self.fcs(x)
        return self.action_bound*self.output(x).double()

class ValidationCriticMLP(MLP):
    def __init__(self, in_feats: int, action_feats: int, hidden_layers: list,
            hidden_activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__(in_feats+action_feats, 1, hidden_layers, hidden_activation, output_activation)

        self.state_shape = (in_feats,)
        self.action_shape = (action_feats,)
    
    def forward(self, x, action):
        x = shape_input_tensor(x.float(), self.state_shape)
        action = shape_input_tensor(action.float(), self.action_shape)

        out = torch.cat([x, action], 1)
        out = self.fcs(out)
        return self.output(out)

########################################################################################################
#                                      stochastic policies                                             #
########################################################################################################
class ValidationSquashedGaussianPolicy(ValidationActorMLP):
    def __init__(self, action_distribution, in_feats: int, action_feats: int, hidden_layers: list, 
            hidden_activation=nn.ReLU, output_activation=nn.Tanh, action_bound=1.05):
        super().__init__(in_feats, action_feats, hidden_layers, hidden_activation, output_activation, action_bound)

        # override output layer to output mean, log_std
        delattr(self, 'output')
        self.mu = nn.Linear(hidden_layers[-1], action_feats)
        self.log_std = nn.Linear(hidden_layers[-1], action_feats)

        # bounds for log_std output
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        
        # for generating stochastic output
        self.action_dist  = action_distribution
    
    # get mean and log_std for action distribution
    def action_mean_log_std(self, state):
        out = self.fcs(state.float())
        # get mean, log_std
        mean = self.mu(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def forward(self, state, deterministic=False):
        state = shape_input_tensor(state, self.state_shape)
        # get action means and log_std
        mean, log_std = self.action_mean_log_std(state)
        if deterministic:
            return mean.double()
        # get squashed gaussian actions
        pi_action = self.action_dist.action_from_dist(mean, log_std, deterministic)
        # scale gaussian actions to action bounds
        pi_action = self.action_bound*pi_action

        return pi_action.double()

    def action_log_prob(self, state):
        mean, log_std = self.action_mean_log_std(state)
        pi_actions, log_probs = self.action_dist.action_log_probs_from_dist(mean, log_std)
        # scale gaussian actions to action bounds
        pi_actions = self.action_bound*pi_actions
        # if pi'(a|s) = k*pi(a|s) => log(pi'(a|s)) = log(k) + log(pi(a|s))
        log_probs = np.log(self.action_bound) + log_probs

        return pi_actions, log_probs
        

class SquashedGaussianPolicy(PolicyNetwork):
    def __init__(self,
            action_distribution,
            channels=[16, 32, 32],
            fc_layers=[1024,512],
            in_channels=4,
            input_shape=(64,64,4),
            action_sz=64,
            action_bound=1.05
        ):
        latent_vec_size = fc_layers[-1]
        
        super().__init__(channels, fc_layers, in_channels, input_shape, (action_sz,), action_bound)
        
        # override the output layer to output mean, log_std
        delattr(self, 'output')
        self.mu = nn.Linear(latent_vec_size, action_sz)
        self.log_std = nn.Linear(latent_vec_size, action_sz)

        # bounds for log_std output
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        # for generating stochastic output
        self.action_dist  = action_distribution
    
    # get mean and log_std for action distribution
    def action_mean_log_std(self, state):
        state = shape_input_tensor(state, self.input_shape)
        feats = self.feats_extractor(state)
        out = self.fc(feats)

        # get mean, log_std
        mean = self.mu(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mean, log_std

    def forward(self, state, deterministic=False):
        mean, log_std = self.action_mean_log_std(state)
        return  self.action_dist.action_from_dist(mean, log_std, deterministic)

    def action_log_prob(self, state):
        mean, log_std = self.action_mean_log_std(state)
        return self.action_dist.action_log_probs_from_dist(mean, log_std)

########################################################################################################
#                                             discrete models                                          #
########################################################################################################
class SoftQNetwork(nn.Module):
    def __init__(self, 
                channels=[16, 32, 32],
                fc_layers=[1024,512,512],
                in_channels=4,
                input_shape=(64,64,4),
                n_actions=64
            ):
        super(SoftQNetwork, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.action_shape = (n_actions,)

        # feature extractor
        self.feats_extractor = IMPALA(in_channels, channels)

        # fc layers
        fc_in_feats = self.feats_extractor.flattened_dim
        fc_out_feats = fc_layers[-1]
        del fc_layers[-1]

        self.fc = MLP(fc_in_feats, fc_out_feats, fc_layers)

        # output 
        self.output = nn.Linear(fc_out_feats, self.n_actions)
    
    def forward(self, x):
        x = shape_input_tensor(x, self.input_shape)
        out = self.feats_extractor(x)
        out = self.fc(out)
        out = self.output(out)

        return out

class DiscreteStochasticPolicy(nn.Module):
    def __init__(self, 
                action_distribution: Categorical,
                channels=[16, 32, 32],
                fc_layers=[1024,512,512],
                in_channels=4,
                input_shape=(64,64,4),
                n_actions=64
            ):
        super(DiscreteStochasticPolicy, self).__init__()

        self.action_dist = action_distribution

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.action_shape = (n_actions,)

        # feature extractor
        self.feats_extractor = IMPALA(in_channels, channels)

        # fc layers
        fc_in_feats = self.feats_extractor.flattened_dim
        fc_out_feats = fc_layers[-1]
        del fc_layers[-1]

        self.fc = MLP(fc_in_feats, fc_out_feats, fc_layers)

        # output 
        self.output = nn.Linear(fc_out_feats, self.n_actions)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, deterministic=False):
        probs = self.action_probs(x)
        actions = self.action_dist.action_from_dist(probs, deterministic)
        return actions

    def action_probs(self, x):
        x = shape_input_tensor(x, self.input_shape)
        x = self.feats_extractor(x)
        x = self.fc(x)
        x = self.output(x)
        # take softmax to generate action probabilities
        action_probs = self.softmax(x)

        return action_probs
    
    def action_log_prob(self, x):
        probs = self.action_probs(x)
        _, log_probs = self.action_dist.action_log_probs_from_dist(probs)
        return probs, log_probs

########################################################################################################
#                                             helper funcs                                             #
########################################################################################################
def shape_input_tensor(tensor, shape):
    """
        Formats the input shape of the tensor appropriately -> adds a batch dimension if there is none
    """
    if len(tensor.shape) == len(shape):
        # single example, add batch dimension
        return tensor.unsqueeze(0)

    else:
        return tensor