import yaml
import torch
import torch.optim as optim

from torch.utils.data import DataLoader 
from training.trajectory_dataset import TrajectoryDataset
from diffusion.diffusion_policy import DiffusionPolicy
from utils.ema import EMA
from training.trainer import Trainer

# train.py is the entry point — it sets everything up:

# Load the config YAML
# Instantiate the dataset and wrap it in a DataLoader (batch size, shuffle, num_workers)
# Instantiate the DiffusionPolicy with the config parameters
# Instantiate the optimizer (AdamW is standard)
# Instantiate EMA with the policy
# Optionally set up a learning rate scheduler (cosine annealing works well)
# Call the trainer

class Train():
    def __init__(self, num_epochs):

        with open("configs/pusht_default.yaml", "r") as f:
            config = yaml.safe_load(f)

        pred_horizon = config["pred_horizon"]
        action_dim = config["action_dim"]
        embed_dim = config["embed_dim"]
        obs_dim = config["obs_dim"]
        obs_horizon = config["obs_horizon"]
        obs_embeded_dim = config["obs_embeded_dim"]
        num_train_timesteps = config["num_train_timesteps"]

        self.dataset = TrajectoryDataset(
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon
        )
        
        self.train_loader = DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=4)

        self.policy = DiffusionPolicy(pred_horizon, action_dim, embed_dim, obs_dim, obs_horizon, obs_embeded_dim, num_train_timesteps)

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=0.001)

        self.ema = EMA(self.policy)

        self.trainer = Trainer(self.policy, self.train_loader, self.optimizer, self.ema, config)
        self.trainer.train(num_epochs)


if __name__ == "__main__":
    num_epochs = 100

    train = Train(num_epochs)

    
