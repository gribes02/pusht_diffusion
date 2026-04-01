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

        normalizer = Normalizer() 

        self.normalized_states = normalizer.normalize_obs(torch.from_numpy(self.states).float())
        self.normalized_actions = normalizer.normalize_actions(torch.from_numpy(self.actions).float())

    def __len__(self):
        return len(self.normalized_states) - self.obs_horizon - self.pred_horizon + 1
    
    def get_valid_start_indices(self):
        valid_indices = []
        for episode_end in self.episode_ends:
            start_idx = episode_end - self.obs_horizon - self.pred_horizon + 1
            if start_idx >= 0:
                valid_indices.append(start_idx)
        return valid_indices
    
    def __getitem__(self, idx):
        valid_start_indices = self.get_valid_start_indices()
        if idx >= len(valid_start_indices):
            raise IndexError("Index out of range for valid start indices.")
        
        start_idx = valid_start_indices[idx]
        obs = self.normalized_states[start_idx : start_idx + self.obs_horizon]
        actions = self.normalized_actions[start_idx + self.obs_horizon : start_idx + self.obs_horizon + self.pred_horizon]

        return obs, actions
    
    def get_episode(self, episode_idx):
        start_idx = self.episode_ends[episode_idx - 1] if episode_idx > 0 else 0
        end_idx = self.episode_ends[episode_idx]
        return self.normalized_states[start_idx:end_idx], self.normalized_actions[start_idx:end_idx]
    
