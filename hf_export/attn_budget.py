"""Budgeted attention — subsystem #1 (per-token key budget from g_t).

The controller's g_t sets each token's ATTENTION SPAN: how far back it may look.
    span_t = span_min + (T - span_min) · g_t          (in key-distance units)
A soft, temperature-annealed mask gates keys beyond the span; a straight-through
hard mask is available for eval. Because a token with LOW g_t literally cannot
reach a distant key, g_t becomes *causally necessary* wherever long-range
attention is required (e.g. resolving a coreference anaphor to a far antecedent).
Paired with a compute cost that pushes g_t down by default (train.py), the model
must raise g_t precisely on hard tokens — which is exactly what faithfulness
(eval/faithfulness.py) checks. This is the make-or-break mechanism (Stage 4).

Differentiable-top-k note: the budget is a per-token key budget realized as a
recency span (top-k by distance). The soft edge is differentiable in g_t; the
hard mask is the straight-through fallback. Shapes: (B, T, d) in/out.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import ControllerSignal


class BudgetedCausalAttention(nn.Module):
    """Causal MHA with a per-token, g_t-driven attention span (#1)."""

    def __init__(self, d_model: int, n_heads: int, *, dropout: float = 0.0,
                 span_min: int = 8, span_temp: float = 4.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.span_min = span_min
        self.temp = span_temp                       # soft-edge temperature (annealable)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None,
                hard: bool = False) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)   # (B,nH,T,dH)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B,nH,T,T)

        # g_t -> per-query span (key-distance budget). No signal => full span.
        g = (signal.g if signal is not None
             else torch.ones(B, T, device=x.device, dtype=x.dtype))    # (B,T)
        span = self.span_min + (T - self.span_min) * g                 # (B,T)

        t_idx = torch.arange(T, device=x.device)
        dist = (t_idx[:, None] - t_idx[None, :]).float()               # (T,T) t-j
        dist = dist[None, :, :]                                        # (1,T,T)
        slack = span[:, :, None] - dist                               # (B,T,T)
        if hard:
            keep = (slack >= 0).float()
            soft = torch.sigmoid(slack / self.temp)
            gate = keep + (soft - soft.detach())                       # straight-through
        else:
            gate = torch.sigmoid(slack / self.temp)                    # (B,T,T) soft span
        log_gate = torch.log(gate.clamp_min(1e-9)).unsqueeze(1)        # (B,1,T,T)

        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        scores = scores + log_gate
        scores = scores.masked_fill(causal, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
        return self.drop(self.proj(out))


if __name__ == "__main__":
    # Smoke test: low g_t restricts span; a far key is unreachable at g=0.
    torch.manual_seed(0)
    B, T, d = 2, 32, 48
    att = BudgetedCausalAttention(d, n_heads=4, span_min=2, temp=1.0)
    x = torch.randn(B, T, d)
    lo = ControllerSignal(g=torch.zeros(B, T), c=torch.zeros(B, T),
                          logits=torch.zeros(B, T, 2))
    hi = ControllerSignal(g=torch.ones(B, T), c=torch.ones(B, T),
                          logits=torch.zeros(B, T, 2))
    out_lo, out_hi = att(x, lo), att(x, hi)
    # Effective reach: with g=0 (span=2) the last query barely sees the past;
    # with g=1 it sees everything. Outputs must differ.
    diff = (out_lo - out_hi).abs().mean().item()
    print(f"attn_budget smoke OK | out {tuple(out_hi.shape)} | "
          f"lo-vs-hi span mean|delta|={diff:.4f} (should be > 0)")
    assert diff > 1e-4, "span budget had no effect"
