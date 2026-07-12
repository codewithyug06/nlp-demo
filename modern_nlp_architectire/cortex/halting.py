from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

class HaltingHead(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(d_model, 1)

    def forward(self, state: torch.Tensor, g: Optional[torch.Tensor]=None, g_bias: float=0.0) -> torch.Tensor:
        logit = self.lin(state).squeeze(-1)
        if g is not None and g_bias != 0.0:
            logit = logit - g_bias * g
        return logit

@dataclass
class PonderInfo:
    p: torch.Tensor
    expected_steps: torch.Tensor
    kl: torch.Tensor

def ponder_probs(halt_logits: torch.Tensor, min_steps: int) -> torch.Tensor:
    N = halt_logits.shape[0]
    lam = torch.sigmoid(halt_logits)
    if min_steps > 1:
        lam = lam.clone()
        lam[:min_steps - 1] = 0.0
    lam = lam.clone()
    lam[N - 1] = 1.0
    one_minus = 1.0 - lam
    cprod = torch.cumprod(one_minus, dim=0)
    cp_excl = torch.cat([torch.ones_like(cprod[:1]), cprod[:-1]], dim=0)
    p = lam * cp_excl
    return p.permute(1, 2, 0).contiguous()

def expected_steps(p: torch.Tensor) -> torch.Tensor:
    n = torch.arange(1, p.shape[-1] + 1, device=p.device, dtype=p.dtype)
    return (p * n).sum(dim=-1)

def kl_geometric(p: torch.Tensor, prior_lambda: float, eps: float=1e-08) -> torch.Tensor:
    N = p.shape[-1]
    n = torch.arange(N, device=p.device, dtype=p.dtype)
    prior = prior_lambda * (1.0 - prior_lambda) ** n
    prior = prior / prior.sum()
    prior = prior.view(*[1] * (p.dim() - 1), N)
    return (p * (torch.log(p + eps) - torch.log(prior + eps))).sum(dim=-1)
if __name__ == '__main__':
    torch.manual_seed(0)
    N, B, T = (5, 2, 8)
    logits = torch.randn(N, B, T)
    p = ponder_probs(logits, min_steps=2)
    assert torch.allclose(p.sum(-1), torch.ones(B, T), atol=1e-05), 'p must sum to 1'
    assert torch.allclose(p[..., 0], torch.zeros(B, T)), 'min-floor: no halt at step 1'
    es = expected_steps(p)
    kl = kl_geometric(p, 0.5)
    print(f'halting smoke OK | E[steps] mean={es.mean():.2f} range=[{es.min():.2f},{es.max():.2f}] | KL mean={kl.mean():.3f} | sum(p)~1={torch.allclose(p.sum(-1), torch.ones(B, T), atol=1e-05)}')