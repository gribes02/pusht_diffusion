import numpy as np
import torch

class DDPMScheduler():
    def __init__(self, num_train_timesteps=1000, beta_schedule="linear", beta_start=0.0001, beta_end=0.02, clip_sample=False,):

        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_sample = clip_sample
        self.betas = self._get_betas()
        self.betas = torch.from_numpy(self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


    def _get_betas(self):
        if self.beta_schedule == "linear":
            betas = np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=np.float32)
        elif self.beta_schedule == "quadratic":
            betas = np.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps, dtype=np.float32) ** 2
        else:
            raise ValueError(f"Unsupported beta schedule: {self.beta_schedule}, use linear or quadratic.")
        
        return betas
    
    def add_noise(self, x0, noise, timesteps) -> torch.Tensor:

        alpha_hat = self.alphas_cumprod[timesteps]       # [B]
        alpha_hat = alpha_hat.view(-1, 1)                # [B, 1] for broadcasting

        xt = alpha_hat**0.5 * x0 + (1 - alpha_hat)**0.5 * noise

        return xt, noise

    def step(self, model_output, timestep, sample) -> torch.Tensor:
        # Implement the DDPM reverse step, returning x_{t-1}
        # Model output is the predicted noise eps_theta
        print("timestep", timestep)
        print(self.alphas_cumprod.size())
        beta = self.betas[timestep]
        alpha_hat_t = self.alphas_cumprod[timestep]
        prev_alpha_hat_t = self.alphas_cumprod[timestep - 1]
        clean_sample = (sample - (1-alpha_hat_t)**0.5 * model_output) / alpha_hat_t**0.5

        if self.clip_sample:
            clean_sample = clean_sample.clamp(-1, 1)

        c0 = (prev_alpha_hat_t**0.5 * beta)/(1 - alpha_hat_t)
        c1 = (alpha_hat_t**0.5 * (1-prev_alpha_hat_t)) / (1 - alpha_hat_t)
        post_mean = c0 * clean_sample +  c1 * sample

        if timestep == 0:
            post_var = 0
        else:
            post_var = ((1 - prev_alpha_hat_t)/(1 - alpha_hat_t) * beta)**0.5

        z = torch.randn_like(sample)
        prev_sample = post_mean + post_var * z

        return prev_sample
        

if __name__ == "__main__":
    ddpm = DDPMScheduler()

    B = 4  # batch size
    timesteps = torch.randint(0, 1000, (B,), dtype=torch.long)  # one random timestep per sample
    x0 = torch.randn(B, 2)   # batch of 2D actions
    noise = torch.randn_like(x0)

    noisy_action, noise = ddpm.add_noise(x0, noise, timesteps)
    
    model_output = noise
    prev_sample = ddpm.step(model_output, timesteps[0], x0)

    print("Previous Sample:", prev_sample)
    print("Timesteps:", timesteps)
    print("Noise:", noise)
    print("Noisy action:", noisy_action)
