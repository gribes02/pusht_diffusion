import torch

class Normalizer():
    
    def __init__(self):
        self.action_mins = torch.tensor([0,0])
        self.action_maxs = torch.tensor([512,512])

        self.obs_mins = torch.tensor([0, 0, 0, 0, 0])
        self.obs_maxs = torch.tensor([512, 512, 512, 512, 2 * 3.141592653589793])

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
    