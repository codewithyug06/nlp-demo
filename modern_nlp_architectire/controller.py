from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ControllerSignal:
    g: torch.Tensor
    c: torch.Tensor
    logits: torch.Tensor

    def detach(self) -> 'ControllerSignal':
        return ControllerSignal(self.g.detach(), self.c.detach(), self.logits.detach())

class Controller(nn.Module):

    def __init__(self, d_model: int, hidden: Optional[int]=None, g_bias: float=0.0):
        super().__init__()
        hidden = hidden or max(16, d_model // 4)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.bias[0] = g_bias

    def forward(self, h: torch.Tensor) -> ControllerSignal:
        logits = self.fc2(F.gelu(self.fc1(h)))
        g = torch.sigmoid(logits[..., 0])
        c = torch.sigmoid(logits[..., 1])
        return ControllerSignal(g=g, c=c, logits=logits)
if __name__ == '__main__':
    torch.manual_seed(0)
    ctrl = Controller(d_model=384)
    h = torch.randn(2, 16, 384)
    sig = ctrl.forward(h)
    assert sig.g.shape == (2, 16) and sig.c.shape == (2, 16)
    assert float(sig.g.min()) >= 0.0 and float(sig.g.max()) <= 1.0
    print(f'controller smoke OK | g mean={sig.g.mean():.3f} c mean={sig.c.mean():.3f} | params={sum((p.numel() for p in ctrl.parameters()))}')