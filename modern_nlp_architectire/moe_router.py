from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .controller import ControllerSignal

class Expert(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc3 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)) * self.fc3(x))

@dataclass
class MoEInfo:
    aux_loss: torch.Tensor
    load: torch.Tensor
    imbalance: float
    mean_k: float

class MoELayer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, n_experts: int=4, k_max: int=2):
        super().__init__()
        assert 1 <= k_max <= n_experts <= 8, 'keep E<=8 and 1<=k_max<=E'
        self.n_experts = n_experts
        self.k_max = k_max
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.last_info: Optional[MoEInfo] = None

    def forward(self, x: torch.Tensor, signal: Optional[ControllerSignal]=None) -> torch.Tensor:
        B, T, d = x.shape
        E, k_max = (self.n_experts, self.k_max)
        xf = x.reshape(B * T, d)
        N = xf.shape[0]
        probs = F.softmax(self.router(xf), dim=-1)
        g = signal.g.reshape(N) if signal is not None else torch.ones(N, device=x.device, dtype=x.dtype)
        k_t = (1 + (k_max - 1) * g).round().long().clamp(1, k_max)
        topv, topi = probs.topk(k_max, dim=-1)
        ranks = torch.arange(k_max, device=x.device)
        keep = ranks[None, :] < k_t[:, None]
        w = topv * keep
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-09)
        outs = torch.stack([e(xf) for e in self.experts], dim=1)
        sel = outs.gather(1, topi[..., None].expand(N, k_max, d))
        y = (w[..., None] * sel).sum(dim=1).reshape(B, T, d)
        top1 = topi[:, 0]
        f = torch.bincount(top1, minlength=E).float() / N
        P = probs.mean(dim=0)
        aux = E * (f * P).sum()
        self.last_info = MoEInfo(aux_loss=aux, load=f.detach(), imbalance=(f.max() * E).item(), mean_k=k_t.float().mean().item())
        return y
if __name__ == '__main__':
    torch.manual_seed(0)
    B, T, d = (2, 16, 48)
    moe = MoELayer(d, d_ff=96, n_experts=4, k_max=3)
    x = torch.randn(B, T, d)
    g = torch.zeros(B, T)
    g[:, ::2] = 1.0
    sig = ControllerSignal(g=g, c=torch.zeros(B, T), logits=torch.zeros(B, T, 2))
    y = moe(x, sig)
    assert y.shape == (B, T, d)
    info = moe.last_info
    print(f'moe smoke OK | out {tuple(y.shape)} | mean_k={info.mean_k:.2f} imbalance={info.imbalance:.2f} (1=balanced) aux={float(info.aux_loss):.3f} load={[round(x, 2) for x in info.load.tolist()]}')