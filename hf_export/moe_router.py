"""Mixture-of-Experts FFN — subsystem #4 (difficulty-scaled expert count).

Standard token-choice MoE (top-k routing over E experts) with ONE twist from the
thesis: the number of active experts per token scales with g_t —
    k_t = round(1 + (k_max − 1) · g_t)  ∈ [1, k_max]
so easy tokens use a single expert (cheap) and hard tokens fan out to more. A
Switch-style load-balancing loss keeps expert usage even.

REDUCTION note (like #3): experts are computed DENSELY (all E experts on all
tokens, then masked-combined) rather than via sparse token dispatch. Routing
statistics and the adaptive expert count are exact and measurable; the FLOP
saving of true sparse dispatch is not yet realized (kept for honesty). E ≤ 8.

Shapes: (B, T, d) in/out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import ControllerSignal


class Expert(nn.Module):
    """One FFN expert. (N, d) -> (N, d)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


@dataclass
class MoEInfo:
    aux_loss: torch.Tensor       # load-balancing loss (scalar)
    load: torch.Tensor           # (E,) fraction of tokens per expert (top-1)
    imbalance: float             # max_load / uniform_load (1.0 = perfectly balanced)
    mean_k: float                # mean active experts per token


class MoELayer(nn.Module):
    """Difficulty-scaled top-k MoE FFN (#4). forward(x, signal)->(B,T,d)."""

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, k_max: int = 2):
        super().__init__()
        assert 1 <= k_max <= n_experts <= 8, "keep E<=8 and 1<=k_max<=E"
        self.n_experts = n_experts
        self.k_max = k_max
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.last_info: Optional[MoEInfo] = None

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None) -> torch.Tensor:
        B, T, d = x.shape
        E, k_max = self.n_experts, self.k_max
        xf = x.reshape(B * T, d)                              # (N, d)
        N = xf.shape[0]
        probs = F.softmax(self.router(xf), dim=-1)           # (N, E)

        # Difficulty-scaled active expert count per token.
        g = (signal.g.reshape(N) if signal is not None
             else torch.ones(N, device=x.device, dtype=x.dtype))
        k_t = (1 + (k_max - 1) * g).round().long().clamp(1, k_max)   # (N,)

        topv, topi = probs.topk(k_max, dim=-1)               # (N, k_max)
        ranks = torch.arange(k_max, device=x.device)
        keep = ranks[None, :] < k_t[:, None]                 # (N, k_max) bool
        w = topv * keep
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # renormalize kept

        # Dense expert compute, then gather the selected experts per token.
        outs = torch.stack([e(xf) for e in self.experts], dim=1)      # (N, E, d)
        sel = outs.gather(1, topi[..., None].expand(N, k_max, d))     # (N, k_max, d)
        y = (w[..., None] * sel).sum(dim=1).reshape(B, T, d)          # (B, T, d)

        # Switch load-balancing loss: f_e (top-1 token fraction) . P_e (mean prob).
        top1 = topi[:, 0]
        f = torch.bincount(top1, minlength=E).float() / N            # (E,)
        P = probs.mean(dim=0)                                         # (E,)
        aux = E * (f * P).sum()
        self.last_info = MoEInfo(aux_loss=aux, load=f.detach(),
                                 imbalance=(f.max() * E).item(),
                                 mean_k=k_t.float().mean().item())
        return y


if __name__ == "__main__":
    # Smoke test: shape ok; hard tokens use more experts; aux>0.
    torch.manual_seed(0)
    B, T, d = 2, 16, 48
    moe = MoELayer(d, d_ff=96, n_experts=4, k_max=3)
    x = torch.randn(B, T, d)
    g = torch.zeros(B, T); g[:, ::2] = 1.0
    sig = ControllerSignal(g=g, c=torch.zeros(B, T), logits=torch.zeros(B, T, 2))
    y = moe(x, sig)
    assert y.shape == (B, T, d)
    info = moe.last_info
    print(f"moe smoke OK | out {tuple(y.shape)} | mean_k={info.mean_k:.2f} "
          f"imbalance={info.imbalance:.2f} (1=balanced) aux={float(info.aux_loss):.3f} "
          f"load={[round(x,2) for x in info.load.tolist()]}")
