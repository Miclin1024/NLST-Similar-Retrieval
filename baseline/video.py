import torch
import attrs
from torchvision import models

@attrs.define(slots=True, init=False)
class ResNetVideoEncoder(torch.nn.Module):
    
    # Video ResNet encoder. 
    # Expect batched input (B, C, D, H, W). Accept PIL image.
    encoder: torch.nn.Module
    
    pretrained: bool
    
    def __init__(self, pretrained=True) -> None:
        super().__init__()
        self.encoder = models.video.r3d_18(
            weights=models.video.R3D_18_Weights.DEFAULT if pretrained else None
        ).to("cuda")
        self.pretrained = pretrained

    @property
    def name(self) -> str:
        return f"r3d_18_{'pretrained' if self.pretrained else 'random'}"
        
    @property
    def description(self) -> str:
        return f"video ResNet encoder with " \
               f"{'pretrained' if self.pretrained else 'random'} weights" \
        
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1, 1)
        return self.encoder(x)
        