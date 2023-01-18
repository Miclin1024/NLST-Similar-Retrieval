import torch
import attrs
from torchvision import models


@attrs.define(slots=True, init=False)
class ResNetSliceWiseEncoder(torch.nn.Module):
    
    # 2D ResNet encoder. 
    # Expect batched input (B, C, H, W). Accept PIL image.
    encoder: torch.nn.Module
    
    pretrained: bool

    def __init__(self, pretrained=True) -> None:
        super().__init__()
        self.encoder = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        ).to("cuda")
        self.pretrained = pretrained

    @property
    def name(self) -> str:
        return f"resnet_2d_{'pretrained' if self.pretrained else 'random'}"

    @property
    def description(self) -> str:
        return f"slice wise ResNet encoder with " \
               f"{'pretrained' if self.pretrained else 'random'} weights + global max pool"
    
    def forward(self, x):
        outputs = []
        for slice in self.slice_wise_generate(x):
            slice = slice.repeat(1, 3, 1, 1)
            outputs.append(self.encoder(slice))
            
        result, _ = torch.max(
            torch.stack(outputs, dim=1),
            dim=1
        )
        
        return result
        
    @staticmethod
    def slice_wise_generate(x):
        num_slices = x.shape[4]
        for i in range(num_slices):
            yield x[:, :, :, i]
    
    
    

