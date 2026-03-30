import torch
import torch.nn as nn

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
    def __init__(self):
        super().__init__()
        

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()













if __name__ == "__main__":
    
    c_in = 64
    c_out = 128
    cond_dim = 256
    B = 4
    T = 125

    features = torch.randn(B, c_in, T)
    cond_vec = torch.randn(B, cond_dim)

    resblock = ResidualConvBlock(c_in, c_out, cond_dim)

    print(resblock(features, cond_vec).shape)
    

