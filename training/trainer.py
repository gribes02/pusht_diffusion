import torch
import torch.nn as nn
import torch.optim as optim
from diffusion.noise_scheduler import DDPMScheduler
from diffusion.diffusion_policy import DiffusionPolicy

class Trainer():
    def __init__(self, policy, train_data_loader, optimizer, ema, config):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = policy
        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.ema = ema
        self.config = config

        self.scheduler = self.policy.noise_scheduler

        self.loss_eq = nn.MSELoss()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for i, data in enumerate(self.train_data_loader):
                observations, actions = data
                observations = observations.to(self.device)
                actions = actions.to(self.device)

                batch_size = observations.shape[0]
                num_train_timesteps = self.scheduler.num_train_timesteps

                t = torch.randint(0, num_train_timesteps, (batch_size,)).to(self.device)
                noise = torch.randn_like(actions).to(self.device)

                noisy_actions, _ = self.scheduler.add_noise(actions, noise, t)

                predicted_noise = self.policy.predict_noise(noisy_actions, t, observations)

                loss = self.loss_eq(predicted_noise, noise)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                self.ema.update(self.policy)

                if i % 50 == 0:
                    print(f"Epoch: {epoch} Loss: {loss.item()}")



            








        