from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

class LayerScale(nn.Module):

    def __init__(self, d_model: int, init: float=0.01):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((d_model,), init))

    def forward(self, sub_out: torch.Tensor) -> torch.Tensor:
        return self.gamma * sub_out

class ResidualScaler(nn.Module):

    def __init__(self, d_model: int, mode: str='vanilla', init: float=0.01):
        super().__init__()
        self.mode = mode
        self.layerscale = LayerScale(d_model, init) if mode == 'layerscale' else None

    def forward(self, sub_out: torch.Tensor) -> torch.Tensor:
        if self.mode == 'vanilla':
            return sub_out
        if self.mode == 'layerscale':
            return self.layerscale(sub_out)
        raise ValueError(f'unknown residual mode: {self.mode}')
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    van = ResidualScaler(16, 'vanilla')
    ls = ResidualScaler(16, 'layerscale', init=0.01)
    assert torch.equal(van(x), x)
    assert torch.allclose(ls(x), 0.01 * x)
    print('residual smoke OK | vanilla no-op; layerscale gamma*out (init 1e-2)')