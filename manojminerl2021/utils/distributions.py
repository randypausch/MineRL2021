import torch
from torch.distributions import Normal
from torch.distributions import Categorical as TorchCategorical

def sum_independent_dims(tensor: torch.Tensor):
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

# Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.
# modified from stable-baselines3
class SquashedGaussianDistribution:
    def __init__(self, action_dim: int, epsilon: float= 1e-6):
        self.distribution = None
        self.action_dim = action_dim
        self.mean = None
        self.log_std = None
        self.epsilon = epsilon
        self.gaussian_actions = None
    
    # create normal distribution from mean and log_std
    def proba_distribution(self, mean: torch.Tensor, log_std: torch.Tensor):
        std = torch.ones_like(mean) * log_std.exp()
        self.distribution = Normal(mean, std)
        return self
    
    # get stochastic action from the distribution
    def action_from_dist(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic=False):
        # update distribution
        self.proba_distribution(mean, log_std)

        if deterministic:
            return self.distribution.mean

        self.gaussian_actions = self.distribution.rsample()
        return torch.tanh(self.gaussian_actions)        

    # Log likelihood for a Gaussian distribution
    def log_prob(self, actions, gaussian_actions=None):
        if gaussian_actions is None:
            eps = torch.finfo(actions.dtype).eps
            actions = torch.clamp(actions, min=-1.0 + eps, max=1.0 - eps)
            gaussian_actions = 0.5 * (actions.log1p() - (-actions).log1p())

        log_probs = sum_independent_dims(self.distribution.log_prob(gaussian_actions))
        # squash correction (SAC paper)
        log_probs -= torch.sum(torch.log(1 - actions ** 2 + self.epsilon), dim=1)
        return log_probs

    # get stochastic actions and log probabilities for actions
    def action_log_probs_from_dist(self, mean: torch.Tensor, log_std: torch.Tensor):
        # get squashed gaussian actions
        pi_actions = self.action_from_dist(mean, log_std)
        # get squashed guassian actions log probs
        log_probs = self.log_prob(pi_actions, self.gaussian_actions)
        return pi_actions, log_probs

class Categorical:
    def __init__(self, n_actions: int, eps: float=1e-8):
        self.n_actions = n_actions
        self.epsilon = eps
        
    # create categorical distribution from probabilities
    def proba_distribution(self, probs: torch.Tensor):
        assert probs.shape[1] == self.n_actions, "no. of probabilities ({}) should match no. of actions ({})".format(probs.shape[1], self.n_actions)

        self.distribution = TorchCategorical(probs)
        return self

    # get stochastic action from the distribution
    def action_from_dist(self, probs: torch.Tensor, deterministic=False):
        # update distribution
        self.proba_distribution(probs)

        if deterministic:
            return torch.argmax(probs, dim=1, keepdim=True)

        actions = self.distribution.sample().view(-1, 1)
        return actions

    # log probabilities from action probabilities
    def log_prob(self, action_probs: torch.Tensor):
        z = (action_probs == 0.0).float() * self.epsilon
        return torch.log(action_probs + z)

    # get stochastic actions and log probabilities for actions
    def action_log_probs_from_dist(self, probs: torch.Tensor):
        # get discrete actions from probabilities
        pi_actions = self.action_from_dist(probs)
        # get action log probs
        log_probs = self.log_prob(probs)

        return pi_actions, log_probs