import torch
import torch.nn as nn
from obs_encoder import ObsEncoder

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim
        

    def forward(self,timesteps) -> torch.Tensor:
        # Timesteps shape [B]

        assert self.embed_dim % 2 == 0, \
        f"embed_dim must be even"

        freq = torch.zeros(self.embed_dim // 2)
        
        i = torch.arange(self.embed_dim // 2)
        freq = 1 / (10000 ** (2*i/self.embed_dim))

        timesteps = timesteps.float()
        embeddings = torch.outer(timesteps, freq) # (B, d/2)

        sin_embed = torch.sin(embeddings)
        cos_embed = torch.cos(embeddings)

        out = torch.cat((sin_embed, cos_embed), dim=1) #Along the embedded dimensions
        
        return out # (B, embed_dim)
    
class FiLM(nn.Module):
    '''The conditioning information (fused timesteps + observation vector) needs to 
        influence the convolutional features at every layer of the U-Net. FiLM does this 
        by learning to predict per-channel scale and shift that get applied to the feature map'''
    
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        
        self.cond_dim = cond_dim
        self.feature_dim = feature_dim

        self.linear = nn.Linear(cond_dim, 2*feature_dim)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features, cond) -> torch.Tensor:
        # features shape [B, feature_dim, T]
        # cond shape [B, cond_dim]

        assert features.shape[1] == self.feature_dim, \
        f"Expected feature_dim={self.feature_dim}, got {features.shape[1]}"    

        assert cond.shape[-1] == self.cond_dim, \
        f"Expected cond_dim={self.cond_dim}, got {cond.shape[-1]}"

        x = self.linear(cond)
        
        gamma, beta = torch.split(x, self.feature_dim, dim=-1) # [B, feature_dim]
        # print("Gamma shape is: ", gamma.shape)
        # print("Beta shape is ", beta.shape)
        gamma = gamma.unsqueeze(dim=-1) # [B, feature_dim, 1]
        beta = beta.unsqueeze(dim=-1)

        return (1 + gamma) * features + beta

class ResidualConvBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        ''' Feature map is of the shape [B, C_in, T] and it would have to output [B,C_out, T]
            Internal flow: Conv1d -> GroupNorm -> Mish
                            FiLM conditioning 
                            Conv1d -> GroupNorm -> Mish
                            Residual Connection '''

        kernel_size = 3
        num_groups = 8

        assert c_out % num_groups == 0, \
        f"Dimensions number of out channel = {c_out} but it must be divisible by {num_groups}"

        self.block1 = nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding= kernel_size // 2),
                                   nn.GroupNorm(num_groups, c_out),
                                   nn.Mish())
        
        self.film = FiLM(cond_dim, c_out) # [B, feature_dim, T]

        self.block2 = nn.Sequential(nn.Conv1d(c_out, c_out, kernel_size=kernel_size, padding= kernel_size // 2),
                                    nn.GroupNorm(num_groups, c_out),
                                    nn.Mish())
        
        self.resize = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, feature_map, cond_vect):
        out_block1 = self.block1(feature_map)
        out_film = self.film(out_block1, cond_vect)
        out_block2 = self.block2(out_film)

        feature_map_resized = self.resize(feature_map)

        return feature_map_resized + out_block2 # [B, c_out, T]

class EncoderBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()

        self.res_block = ResidualConvBlock(c_in, c_out, cond_dim)
        self.downsample = nn.Conv1d(c_out, c_out, kernel_size=3, stride=2, padding=1)

    def forward(self, feature_map, cond_vect): 

        feature_map = self.res_block(feature_map, cond_vect)
        downsample = self.downsample(feature_map)

        return downsample, feature_map
    
class Bottleneck(nn.Module):
    def __init__(self, c_in, cond_dim):
        super().__init__()

        self.residual_block1 = ResidualConvBlock(c_in, c_in, cond_dim)
        self.residual_block2 = ResidualConvBlock(c_in, c_in, cond_dim)
        
    def forward(self, feature_map, cond_vect):
        out1 = self.residual_block1(feature_map, cond_vect)
        out2 = self.residual_block2(out1, cond_vect)

        return out2


class DecoderBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.residual_block1 = ResidualConvBlock(c_in, c_out, cond_dim)
        self.residual_block2 = ResidualConvBlock(c_out, c_out, cond_dim)

    def forward(self, skip_connection, feature_map, cond_vect):

        out_upsample = self.upsample(feature_map)
        out_connection = torch.cat((skip_connection, out_upsample), dim=1)
        out_res1 = self.residual_block1(out_connection, cond_vect)
        out_res2 = self.residual_block2(out_res1, cond_vect)

        return out_res2
    
class ConditionalVector(nn.Module):
    def __init__(self, embed_dim, obs_dim, obs_horizon, obs_embeded_dim):
        super().__init__()

        self.sin_embed = SinusoidalEmbedding(embed_dim) # [B, embed_dim]
        self.obs_encoder = ObsEncoder(obs_dim, obs_horizon, obs_embeded_dim) # [B, obs_embed_dim]

        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), 
                                 nn.Mish(), 
                                 nn.Linear(embed_dim * 2, embed_dim),
                                 nn.Mish())

        self.resize = nn.Linear(obs_embeded_dim, embed_dim) if obs_embeded_dim != embed_dim else nn.Identity()
    
    def forward(self, timesteps, obs): 
        
        sin_embed_out = self.sin_embed(timesteps)
        sin_embed_out = self.mlp(sin_embed_out)
        obs_encoder_out = self.obs_encoder(obs)

        obs_encoder_out = self.resize(obs_encoder_out)
        
        return sin_embed_out + obs_encoder_out


class UNet(nn.Module):
    def __init__(self, embed_dim, obs_dim, obs_horizon, obs_embeded_dim, base_dim=256, action_dim=2):
        super().__init__()

        multipliers = [1, 2, 4]

        cond_dim = embed_dim

        self.resize = nn.Conv1d(action_dim, base_dim, kernel_size=1)

        dims = [base_dim * m for m in multipliers] # [base_dim * 1, base_dim * 2, base_dim * 4]

        self.encoders = nn.ModuleList(EncoderBlock(dims[i], dims[i + 1], cond_dim) for i in range(len(dims) - 1))
        self.decoders = nn.ModuleList(DecoderBlock(dims[i + 1] * 2, dims[i], cond_dim) for i in reversed(range(len(dims) - 1)))

        in_bottleneck = dims[-1]
        self.bottleneck = Bottleneck(in_bottleneck, cond_dim)

        self.cond_vect = ConditionalVector(cond_dim, obs_dim, obs_horizon, obs_embeded_dim)

        self.output_proj = nn.Conv1d(base_dim, action_dim, kernel_size=1)

        

    def forward(self, noisy_action, timesteps, obs):

        skip_connections = []

        noisy_action = noisy_action.transpose(2, 1) # [B, action_dim, pred_horizon]

        x = self.resize(noisy_action)

        cond_vect = self.cond_vect(timesteps, obs)

        for encoder in self.encoders:
            x, skip_connection = encoder(x, cond_vect)
            skip_connections.append(skip_connection)

        x = self.bottleneck(x, cond_vect)

        for decoder in self.decoders:
            skip_connection = skip_connections.pop()
            x = decoder(skip_connection, x, cond_vect)

        out = self.output_proj(x)

        return out.transpose(2,1)         
        
if __name__ == "__main__":
    
    c_in = 64
    c_out = 128
    cond_dim = 256
    B = 4
    T = 125
    obs_dim = 12
    obs_horizon = 3

    pred_horizon = 12
    action_dim = 2

    features = torch.randn(B, c_in, T)
    cond_vec = torch.randn(B, cond_dim)

    noisy_action = torch.randn((B, pred_horizon, action_dim))
    timestep = torch.randn((B,))
    obs = torch.randn((B, obs_horizon, obs_dim))

    u_net = UNet(cond_dim, obs_dim, obs_horizon, obs_embeded_dim=256)    

    out = u_net(noisy_action, timestep, obs)
    print(out)
    print(out.shape) # (B, pred_horizon, action_dim)

