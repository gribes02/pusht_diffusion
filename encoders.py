import torch 
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class StateEncoder(nn.Module):

    def __init__(self, in_features, embed_dim=256, hidden_dims=256):
        super().__init__()
        
        # x (B, 2)

        self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dims),
                nn.GELU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.GELU(),
                nn.Linear(hidden_dims, embed_dim)
                )

        def forward(self, x):
            return self.net(x)


class ImageEncoder(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.resnet18_model = resnet18(weights=weights)
        # Removing the last fc layer, leading to a shape of (B, 512)
        self.backbone = nn.Sequential(*list(self.resnet18_model.children())[:-1])

        resnet_output_dim = self.resnet18_model.fc.in_features
        self.projection = nn.Linear(resnet_output_dim, embed_dim)

    def forward(self, x):
        # x: [B, C, W, H] 
        features = self.backbone(x) # (B, 512, 1, 1)

        flattened = torch.flatten(features, 1) # (B, 512)

        embeddings = self.projection(flattened) # (B, 256)

        return embeddings


def main():
    batch_size = 16
    image = torch.randn(batch_size, 3, 96, 96) 
    print(image.shape)
    image_encoder = ImageEncoder()
    image_features = image_encoder(image)

    print(image_features.shape)


if __name__=="__main__":
    print("bro really debugging rn")
    main()
    
