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
    def __init__(self, cond_dim, feature_dim):
        super().__init()
        
