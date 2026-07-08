"""Transformer block + the single wiring point for ALL subsystems.

`block.py` is where every subsystem hangs off ONE controller signal. Right now
(Stage 2) the block computes the standard attn+MLP and accepts the shared
`(g_t, c_t)` signal but CONSUMES NOTHING — a deliberate logged no-op so we can
prove the plumbing is neutral before any subsystem gets power.

As later stages land, each subsystem reads `signal` HERE:
    #1 attn_budget -> key budget from g_t     #7 residual   -> structured residual
    #5 halting     -> per-token halting        #9 latent_loop-> refine hard tokens
    #4 moe_router  -> expert count from g_t     #6 mem        -> gated read/write
The signal object never forks per subsystem — that is the whole coupling thesis.

Shapes annotated as (B, T, d).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_budget import BudgetedCausalAttention
from .controller import ControllerSignal
from .moe_router import MoELayer
from .residual import ResidualScaler


class CausalSelfAttention(nn.Module):
    """Explicit multi-head causal attention. (B, T, d) -> (B, T, d)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None) -> torch.Tensor:
        
        B, T, d = x.shape
        qkv = self.qkv(x)                                   
        q, k, v = qkv.chunk(3, dim=-1)                       
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)   
        return self.drop(self.proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward. (B, T, d) -> (B, T, d)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))       


class TransformerBlock(nn.Module):
    """One block. Norm placement (#7), budgeted attention (#1), structured
    residual (#7), and MoE FFN (#4) are all config-selected. forward(h, signal)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *,
                 dropout: float = 0.0, norm: str = "pre",
                 attn_budget: bool = False, span_min: int = 8, span_temp: float = 4.0,
                 residual: str = "vanilla", moe: bool = False,
                 moe_experts: int = 4, moe_k_max: int = 2, moe_expert_ff: int = 0):
        super().__init__()
        self.norm = norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if attn_budget:                                   
            self.attn: nn.Module = BudgetedCausalAttention(
                d_model, n_heads, dropout=dropout, span_min=span_min, span_temp=span_temp)
        else:
            self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.is_moe = moe
        if moe:                                           
            self.mlp: nn.Module = MoELayer(
                d_model, moe_expert_ff or d_ff, moe_experts, moe_k_max)
        else:
            self.mlp = MLP(d_model, d_ff, dropout)
        self.res_attn = ResidualScaler(d_model, residual)  
        self.res_mlp = ResidualScaler(d_model, residual)

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None) -> torch.Tensor:
        if self.norm == "pre":
            x = x + self.res_attn(self.attn(self.ln1(x), signal))
            x = x + self.res_mlp(self.mlp(self.ln2(x), signal))
        elif self.norm == "post":
            x = self.ln1(x + self.res_attn(self.attn(x, signal)))
            x = self.ln2(x + self.res_mlp(self.mlp(x, signal)))
        else:
            raise ValueError(f"unknown norm placement: {self.norm}")
        return x


if __name__ == "__main__":
    
    torch.manual_seed(0)
    blk = TransformerBlock(d_model=384, n_heads=6, d_ff=1536)
    h = torch.randn(2, 16, 384)
    out_none = blk(h)
    from .controller import Controller
    sig = Controller(384).forward(h)
    out_sig = blk(h, sig)
    assert out_none.shape == h.shape == out_sig.shape
    
    assert torch.allclose(out_none, out_sig), "signal is not inert!"
    print(f"block smoke OK | out {tuple(out_sig.shape)} | signal inert: "
          f"{torch.allclose(out_none, out_sig)}")
