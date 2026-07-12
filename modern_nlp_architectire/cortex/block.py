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

from .pos import apply_rotary_emb

from .attn_budget import BudgetedCausalAttention
from .controller import ControllerSignal
from .moe_router import MoELayer
from .residual import ResidualScaler


class CausalSelfAttention(nn.Module):
    """Explicit multi-head causal attention. (B, T, d) -> (B, T, d)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None, dropout: float = 0.0, sliding_window: Optional[int] = None, quantize_kv: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide n_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.d_head = d_model // n_heads
        self.sliding_window = sliding_window
        self.quantize_kv = quantize_kv
        
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None,
                use_cache: bool = False,
                past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                freqs_cis: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        
        B, T_q, d = x.shape
        q = self.wq(x).view(B, T_q, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T_q, self.n_kv_heads, self.d_head)
        v = self.wv(x).view(B, T_q, self.n_kv_heads, self.d_head)
        
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
            
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            if self.quantize_kv:
                # Dynamic Dequantization (very simple implementation for demonstration)
                # Assumes we stored them scaled by 127.0
                past_k = past_k.float() / 127.0
                past_v = past_v.float() / 127.0
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if use_cache:
            if self.quantize_kv:
                # Dynamic Quantization
                store_k = (k * 127.0).round().clamp(-128, 127).to(torch.int8)
                store_v = (v * 127.0).round().clamp(-128, 127).to(torch.int8)
                present_key_value = (store_k, store_v)
            else:
                present_key_value = (k, v)
        else:
            present_key_value = None

        if self.n_rep > 1:
            k = torch.repeat_interleave(k, repeats=self.n_rep, dim=1)
            v = torch.repeat_interleave(v, repeats=self.n_rep, dim=1)

        # Flash Attention (SDPA)
        is_causal = past_key_value is None
        attn_mask = None
        
        # Sliding Window Attention Support
        if self.sliding_window is not None and is_causal:
            is_causal = False
            T_kv = k.shape[2]
            # Create a boolean sliding window mask
            attn_mask = torch.tril(torch.ones(T_q, T_kv, dtype=torch.bool, device=q.device))
            attn_mask = torch.triu(attn_mask, diagonal=-self.sliding_window)

        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0, 
            is_causal=is_causal
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T_q, d)   
        return self.drop(self.proj(out)), present_key_value


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class MLP(nn.Module):
    """Position-wise feed-forward. (B, T, d) -> (B, T, d)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc3 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None) -> torch.Tensor:
        return self.drop(self.fc2(F.silu(self.fc1(x)) * self.fc3(x)))       


class TransformerBlock(nn.Module):
    """One block. Norm placement (#7), budgeted attention (#1), structured
    residual (#7), and MoE FFN (#4) are all config-selected. forward(h, signal)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, *,
                 dropout: float = 0.0, norm: str = "pre",
                 attn_budget: bool = False, span_min: int = 8, span_temp: float = 4.0,
                 residual: str = "vanilla", moe: bool = False,
                 moe_experts: int = 4, moe_k_max: int = 2, moe_expert_ff: int = 0,
                 sliding_window: Optional[int] = None, quantize_kv: bool = False):
        super().__init__()
        self.norm = norm
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        if attn_budget:                                   
            self.attn: nn.Module = BudgetedCausalAttention(
                d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, span_min=span_min, span_temp=span_temp,
                sliding_window=sliding_window, quantize_kv=quantize_kv)
        else:
            self.attn = CausalSelfAttention(d_model, n_heads, n_kv_heads, dropout, sliding_window=sliding_window, quantize_kv=quantize_kv)
        self.is_moe = moe
        if moe:                                           
            self.mlp: nn.Module = MoELayer(
                d_model, moe_expert_ff or d_ff, moe_experts, moe_k_max)
        else:
            self.mlp = MLP(d_model, d_ff, dropout)
        self.res_attn = ResidualScaler(d_model, residual)  
        self.res_mlp = ResidualScaler(d_model, residual)

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None,
                use_cache: bool = False,
                past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                freqs_cis: Optional[torch.Tensor] = None):
        
        present_key_value = None
        if self.norm == "pre":
            attn_out, present_key_value = self.attn(self.ln1(x), signal, use_cache=use_cache, past_key_value=past_key_value, freqs_cis=freqs_cis)
            x = x + self.res_attn(attn_out)
            x = x + self.res_mlp(self.mlp(self.ln2(x), signal))
        elif self.norm == "post":
            attn_out, present_key_value = self.attn(x, signal, use_cache=use_cache, past_key_value=past_key_value, freqs_cis=freqs_cis)
            x = self.ln1(x + self.res_attn(attn_out))
            x = self.ln2(x + self.res_mlp(self.mlp(x, signal)))
        else:
            raise ValueError(f"unknown norm placement: {self.norm}")
            
        if use_cache:
            return x, present_key_value
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
