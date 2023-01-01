import torch
import attrs
from torchvision import models

@attrs.define(slots=True, init=False)
class ResNetSliceWiseEncoder(torch.nn.Module):
    
    # 2D ResNet encoder. 
    # Expect batched input (B, C, H, W). Accept PIL image.
    resnet: torch.nn.Module
    
    def __init__(self) -> None:
        super().__init__()
        self.resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        ).to("cuda")
    
    def forward(self, x):
        outputs = []
        for slice in self.slice_wise_generate(x):
            slice = slice.repeat(1, 3, 1, 1)
            outputs.append(self.resnet(slice))
            
        return torch.stack(outputs, dim=1)
        
    @staticmethod
    def slice_wise_generate(x):
        num_slices = x.shape[4]
        for i in range(num_slices):
            yield x[:, :, :, i]
    
    
    

