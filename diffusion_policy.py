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
        # x: (B, action_dim, action_horizon)
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

        return out # (B, out_channels, horizon)


class DownStage(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim) -> None:
        super().__init__()
        
        self.res_blocks = nn.ModuleList([ResidualBlock(in_channels, out_channels // 2, cond_dim),
                                         ResidualBlock(out_channels // 2, out_channels, cond_dim)])

        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x, cond_vect):
        
        for block in self.res_blocks:
            x = block(x, cond_vect)
            
        skip = x 

        out = self.downsample(x)

        return out, skip

class UpStage(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, cond_dim) -> None:
        super().__init__()

        sum_channels = skip_channels + in_channels
        
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.blocks = nn.ModuleList([ResidualBlock(sum_channels, sum_channels // 2, cond_dim),
                                     ResidualBlock(sum_channels // 2, out_channels, cond_dim)])

    def forward(self, x, skip, cond_vect):
        # x = (B, in_channels, horizon)
        # skip = (B, skip_channels, horizon*2)

        x = self.upsample(x)
 
        x = torch.cat((x, skip), dim=1)

        for block in self.blocks:
            x = block(x, cond_vect)

        return x 

class UNet(nn.Module):

    def __init__(self, action_dim, cond_dim, channel_list, kernel_size=3, num_groups=2):
        super().__init__()

        self.down_stages = nn.ModuleList([DownStage(channel_list[0], channel_list[1], cond_dim),
                                          DownStage(channel_list[1], channel_list[2], cond_dim)])

        self.bottleneck = nn.ModuleList([ResidualBlock(channel_list[2], channel_list[2], cond_dim),
                                         ResidualBlock(channel_list[2], channel_list[2], cond_dim)])

        self.up_stages = nn.ModuleList([UpStage(channel_list[2], channel_list[1], channel_list[2], cond_dim),
                                        UpStage(channel_list[1], channel_list[0], channel_list[1], cond_dim)])

        self.projection = nn.Conv1d(channel_list[0], action_dim, kernel_size=1)

    def forward(self, x, cond_vect):
        # x: (B, C, H)

        skips = []

        for down in self.down_stages:
            x, skip = down(x, cond_vect)
            skips.append(skip)

        for block in self.bottleneck:
            x = block(x, cond_vect)
        
        skips = skips[::-1]
        for i, up_stage in enumerate(self.up_stages):
            x = up_stage(x, skips[i], cond_vect)

        out = self.projection(x)

        return out
        

if __name__=="__main__":

    batch_size = 4
    action_channel = 2
    action_horizon = 16
    action_dim = 2
    cond_dim = 256
    channel_list = [2, 8, 32]

    cond_vect = torch.randn(batch_size, cond_dim) 

    u_net = UNet(action_dim, cond_dim, channel_list)

    noisy_actions = torch.randn(batch_size, action_dim, action_horizon)

    clean_action = u_net(noisy_actions, cond_vect)

    print(clean_action.shape)




