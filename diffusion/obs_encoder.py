import torch
import torch.nn as nn

class ObsEncoder(nn.Module):
    '''Encodes the observation window into a single conditioning vector
        Input: [B, T_obs, obs_dim]
        Output: [B, T_obs_embeded_dim]'''
    
    def __init__(self, obs_dim, obs_horizon, obs_embeded_dim):
        super().__init__()

        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.obs_embeded_dim = obs_embeded_dim
        self.flatten1 = nn.Flatten(1, -1) # Flattens to [B, T_obs * obs_dim]
        self.mlp = nn.Sequential(nn.Linear(self.obs_dim * self.obs_horizon, self.obs_embeded_dim),
                                 nn.LayerNorm(self.obs_embeded_dim),
                                 nn.Mish(),
                                 nn.Linear(self.obs_embeded_dim, self.obs_embeded_dim),
                                 nn.LayerNorm(self.obs_embeded_dim),
                                 nn.Mish(),
                                 nn.Linear(self.obs_embeded_dim, self.obs_embeded_dim),
                                 nn.LayerNorm(self.obs_embeded_dim),
                                 nn.Mish())
        
    def forward(self, obs) -> torch.Tensor:
        # Input: [B, T_obs, obs_dim]
        assert obs.shape[-2] == self.obs_horizon, \
        f"Expected obs_horizon={self.obs_horizon}, got {obs.shape[-2]}"
        assert obs.shape[-1] == self.obs_dim
        f"Expected obs_dim={self.obs_dim}, got {obs.shape[-1]}"
        
        obs_flat = self.flatten1(obs)
        out = self.mlp(obs_flat)

        return out 
