import torch 
import torch.nn as nn
from unet import UNet
from noise_scheduler import DDPMScheduler
from obs_encoder import ObsEncoder


class DiffusionPolicy(nn.Module):
    def __init__(self, pred_horizon, action_dim, embed_dim, obs_dim, obs_horizon, obs_embeded_dim, num_train_timesteps=1000, beta_schedule="linear", beta_start=0.0001, beta_end=0.02, clip_sample=False):
        super().__init__()

        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.num_train_timesteps = num_train_timesteps

        self.unet = UNet(embed_dim, obs_dim, obs_horizon, obs_embeded_dim)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps, beta_schedule, beta_start, beta_end, clip_sample)
        
    def predict_noise(self, noisy_actions, timesteps, obs):
        return self.unet(noisy_actions, timesteps, obs)

    def generate_actions(self, obs):
        batch_size = obs.shape[0]
        noisy_action = torch.randn(batch_size, self.pred_horizon, self.action_dim)

        for i in reversed(range(self.num_train_timesteps)):
            timesteps = torch.full((batch_size,), i)
            predicted_noise = self.unet(noisy_action, timesteps, obs)

            prev_sample = self.noise_scheduler.step(predicted_noise, timesteps, noisy_action)

            noisy_action = prev_sample
            








