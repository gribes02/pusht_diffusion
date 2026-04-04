import zarr
import torch
import numpy as np
from normalizer import Normalizer

class TrajectoryDataset:
    def __init__(self, data_path='pusht_data/', pred_horizon=10, obs_horizon=10):
        self.dataset = zarr.open(data_path, mode="r")
        self.states = self.dataset["data"]["state"]      # (25650, 5)
        self.actions = self.dataset["data"]["action"]    # (25650, 2)
        self.episode_ends = self.dataset["meta"]["episode_ends"]  # (206,)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

        self.valid_indices = self.get_valid_indices()

        normalizer = Normalizer() 

        self.normalized_states = normalizer.normalize_obs(torch.from_numpy(self.states).float())
        self.normalized_actions = normalizer.normalize_actions(torch.from_numpy(self.actions).float())

    def __len__(self):
        return len(self.valid_indices)
    
    def get_valid_indices(self):
        valid_indices = []
        for i, episode_end in enumerate(self.episode_ends):
            episode_start = self.episode_ends[i - 1] if i > 0 else 0
            t_start = episode_start + self.obs_horizon - 1
            t_end = episode_end - self.pred_horizon
            for t in range(t_start, t_end):
                valid_indices.append(t)
        return valid_indices
    
    def __getitem__(self, idx):
        if idx >= len(self.valid_indices):
            raise IndexError("Index out of range for valid start indices.")
        
        start_idx = self.valid_indices[idx]
        obs = self.normalized_states[start_idx - self.obs_horizon + 1 : start_idx + 1]  # (obs_horizon, state_dim)
        actions = self.normalized_actions[start_idx : start_idx + self.pred_horizon] # (pred_horizon, action_dim)

        return obs, actions
    
    def get_episode(self, episode_idx):
        start_idx = self.episode_ends[episode_idx - 1] if episode_idx > 0 else 0
        end_idx = self.episode_ends[episode_idx]
        return self.normalized_states[start_idx:end_idx], self.normalized_actions[start_idx:end_idx]
    
