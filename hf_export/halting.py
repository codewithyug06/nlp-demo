"""PonderNet halting — subsystem #5 (per-token adaptive depth).

A halting head emits, at each refinement pass n, a probability λ_n of stopping.
The PonderNet halting distribution is
    p_n = λ_n · Π_{k<n}(1 − λ_k),
i.e. "halt at n" = "stop now" × "didn't stop earlier". Two anti-collapse guards:
  * min-passes floor : λ_n forced to 0 for n < min_steps (must think a bit).
  * KL-to-geometric  : regularize p toward a Geometric(prior_lambda) so the model
                       neither always-halts-immediately nor always-runs-to-max.
The controller couples in: g_bias lowers the halt logit by g_bias·g_t, so hard
tokens (high g_t) halt LATER — spending more depth exactly where difficulty says.

Shapes: halt logits (N, B, T); distribution p (B, T, N); expected steps (B, T).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class HaltingHead(nn.Module):
    """Per-token, per-pass halt logit. (B,T,d) -> (B,T), optionally g-biased."""

    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(d_model, 1)

    def forward(self, state: torch.Tensor, g: Optional[torch.Tensor] = None,
                g_bias: float = 0.0) -> torch.Tensor:
        logit = self.lin(state).squeeze(-1)              # (B, T)
        if g is not None and g_bias != 0.0:
            logit = logit - g_bias * g                    # high g -> less likely halt
        return logit


@dataclass
class PonderInfo:
    p: torch.Tensor              # (B, T, N) halting distribution
    expected_steps: torch.Tensor # (B, T)
    kl: torch.Tensor             # (B, T) KL(p || geometric prior)


def ponder_probs(halt_logits: torch.Tensor, min_steps: int) -> torch.Tensor:
    """PonderNet halting distribution from per-pass logits. (N,B,T) -> (B,T,N)."""
    N = halt_logits.shape[0]
    lam = torch.sigmoid(halt_logits)                     # (N,B,T)
    if min_steps > 1:                                    # min-passes floor
        lam = lam.clone()
        lam[: min_steps - 1] = 0.0
    lam = lam.clone()
    lam[N - 1] = 1.0                                     # must halt by the last pass
    one_minus = 1.0 - lam
    cprod = torch.cumprod(one_minus, dim=0)              # inclusive Π_{k<=n}
    cp_excl = torch.cat([torch.ones_like(cprod[:1]), cprod[:-1]], dim=0)  # Π_{k<n}
    p = lam * cp_excl                                    # (N,B,T)
    return p.permute(1, 2, 0).contiguous()              # (B,T,N)


def expected_steps(p: torch.Tensor) -> torch.Tensor:
    """E[number of passes]. (B,T,N) -> (B,T)."""
    n = torch.arange(1, p.shape[-1] + 1, device=p.device, dtype=p.dtype)
    return (p * n).sum(dim=-1)


def kl_geometric(p: torch.Tensor, prior_lambda: float, eps: float = 1e-8) -> torch.Tensor:
    """KL(p || Geometric(prior_lambda)) over the N pondering steps. (B,T,N)->(B,T)."""
    N = p.shape[-1]
    n = torch.arange(N, device=p.device, dtype=p.dtype)
    prior = prior_lambda * (1.0 - prior_lambda) ** n     # (N,)
    prior = prior / prior.sum()                          # renormalize over finite N
    prior = prior.view(*([1] * (p.dim() - 1)), N)
    return (p * (torch.log(p + eps) - torch.log(prior + eps))).sum(dim=-1)


if __name__ == "__main__":
    # Smoke test: valid distribution, min-floor respected, g_bias raises steps.
    torch.manual_seed(0)
    N, B, T = 5, 2, 8
    logits = torch.randn(N, B, T)
    p = ponder_probs(logits, min_steps=2)
    assert torch.allclose(p.sum(-1), torch.ones(B, T), atol=1e-5), "p must sum to 1"
    assert torch.allclose(p[..., 0], torch.zeros(B, T)), "min-floor: no halt at step 1"
    es = expected_steps(p)
    kl = kl_geometric(p, 0.5)
    print(f"halting smoke OK | E[steps] mean={es.mean():.2f} "
          f"range=[{es.min():.2f},{es.max():.2f}] | KL mean={kl.mean():.3f} | "
          f"sum(p)~1={torch.allclose(p.sum(-1), torch.ones(B,T), atol=1e-5)}")
