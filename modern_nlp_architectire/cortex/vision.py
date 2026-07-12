import torch
import torch.nn as nn

class CortexVisionEncoder(nn.Module):

    def __init__(self, d_model: int, image_size: int=224, patch_size: int=16, channels: int=3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels=channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.size(0)
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x