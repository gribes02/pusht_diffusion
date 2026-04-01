import torch

class Normalizer():
    
    def __init__(self):
        self.action_mins = torch.tensor([12, 25])
        self.action_maxs = torch.tensor([511,511])

        self.obs_mins = torch.tensor([1.3456424e+01, 3.2938293e+01, 5.7471767e+01, 1.0827995e+02, 2.1559125e-04])
        self.obs_maxs = torch.tensor([496.14618, 510.9579, 439.9153, 485.6641, 6.2830877])

    def normalize_actions(self, actions):
        # Normalize to [-1, 1]
        normalized = 2 * (actions - self.action_mins) / (self.action_maxs - self.action_mins) - 1
        return normalized
    
    def denormalize_actions(self, normalized_actions):
        # Denormalize from [-1, 1] back to original range
        denormalized = (normalized_actions + 1) / 2 * (self.action_maxs - self.action_mins) + self.action_mins
        return denormalized
    
    def normalize_obs(self, obs):
        normalized = 2 * (obs - self.obs_mins) / (self.obs_maxs - self.obs_mins) - 1
        return normalized
    
    def denormalize_obs(self, normalized_obs):
        denormalized = (normalized_obs + 1) / 2 * (self.obs_maxs - self.obs_mins) + self.obs_mins
        return denormalized
    