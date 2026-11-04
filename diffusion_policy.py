import torch
import torch.nn as nn
import math

def sinusoidal_embedding(t, embed_dim=256, max_period=10000):

    half_dim = embed_dim // 2

    # Frequencies 
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)

    # t x Frequencies
    emb = t[:, None] * emb[None, :]

    # sin and cos
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

    return emb

# U-Net inputs (noisy action sequence (B, horizon, action_dim)
                # These actions will have to have a channel dimension for 1D Conv
# U-Net outputs (actions to send to the robot)  
class FiLM(nn.Module):

    def __init__(self, cond_dim, num_channels):
        super().__init__()

        self.num_channels = num_channels
        self.linear = nn.Linear(cond_dim, 2*num_channels)

    def forward(self, cond_vect):

        # cond_vect: (B, cond_dim)
        gammaBeta = self.linear(cond_vect) # (B, 2*num_channels)

        gamma_raw = gammaBeta[:, self.num_channels:]
        beta = gammaBeta[:, :self.num_channels]

        gamma = 1 + gamma_raw

        return gamma, beta

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, num_groups=2): 

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # L_out = (L_in + 2 x padding - kernel_size + 1 - 1) / 2 + 1 

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.film1 = FiLM(cond_dim, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.film2 = FiLM(cond_dim, out_channels)
        self.mish1 = nn.Mish()
        self.mish2 = nn.Mish()
        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, cond_vect):
        # x: (B, action_horizon, action_dim)
        x = x.transpose(1,2) # (B, action_dim, action_horizon)
        cache = x

        x = self.conv1(x) # (B, out_channels, action_horizon)
        x = self.norm1(x)
        gamma, beta = self.film1(cond_vect) 
        gamma = gamma.unsqueeze(-1) 
        beta = beta.unsqueeze(-1) # (B, out_channels, 1)
        x = gamma * x + beta # (B, out_channels, action_horizon)
        x = self.mish1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        gamma, beta = self.film2(cond_vect)
        gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)
        x = gamma * x + beta
        x = self.mish2(x)

        if self.in_channels != self.out_channels:
            cache = self.projection(cache)

        out = x + cache

        return out.transpose(1,2) # (B, horizon, out_channels)

class UNet(nn.Module):

    def __init__(self, action_dim, cond_dim, channel_list, kernel_size=3, num_groups=2):
        super().__init__()

if __name__=="__main__":

    batch_size = 4
    action_dim = 2
    cond_dim = 16
    action_horizon = 16
    out_channels = 16
    num_groups = 2

    cond_vect = torch.randn(batch_size, cond_dim)
    actions = torch.randn(batch_size, action_horizon, action_dim)

    res_block = ResidualBlock(action_dim, out_channels, cond_dim)

    out = res_block(actions, cond_vect)
    print(out.shape) # (batch_size, out_channels, action_horizon) 










