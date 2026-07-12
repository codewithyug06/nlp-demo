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

from .pos import apply_rotary_emb

from .controller import ControllerSignal


class BudgetedCausalAttention(nn.Module):
    """Causal MHA with a per-token, g_t-driven attention span (#1)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None, dropout: float = 0.0,
                 span_min: int = 8, span_temp: float = 4.0, sliding_window: Optional[int] = None, quantize_kv: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide n_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.d_head = d_model // n_heads
        self.span_min = span_min
        self.temp = span_temp                       
        self.sliding_window = sliding_window
        self.quantize_kv = quantize_kv
        
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                signal: Optional[ControllerSignal] = None,
                hard: bool = False,
                use_cache: bool = False,
                past_key_value: Optional[tuple] = None,
                freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor | tuple[torch.Tensor, Optional[tuple]]:
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
                past_k = past_k.float() / 127.0
                past_v = past_v.float() / 127.0
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        if use_cache:
            if self.quantize_kv:
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
            
        T_kv = k.shape[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   

        
        g = (signal.g if signal is not None
             else torch.ones(B, T_q, device=x.device, dtype=x.dtype))    
        span = self.span_min + (T_kv - self.span_min) * g                 

        start_pos = T_kv - T_q
        t_idx_q = torch.arange(start_pos, start_pos + T_q, device=x.device)
        t_idx_k = torch.arange(T_kv, device=x.device)
        dist = (t_idx_q[:, None] - t_idx_k[None, :]).float()               
        dist = dist[None, :, :]                                        
        slack = span[:, :, None] - dist                               
        if hard:
            keep = (slack >= 0).float()
            soft = torch.sigmoid(slack / self.temp)
            gate = keep + (soft - soft.detach())                       
        else:
            gate = torch.sigmoid(slack / self.temp)                    
        log_gate = torch.log(gate.clamp_min(1e-9)).unsqueeze(1)        

        causal = (t_idx_q[:, None] < t_idx_k[None, :])
        
        if self.sliding_window is not None:
            # Also mask out things outside the sliding window
            out_of_window = (t_idx_q[:, None] - t_idx_k[None, :]) > self.sliding_window
            causal = causal | out_of_window
            
        scores = scores + log_gate
        scores = scores.masked_fill(causal, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T_q, d)
        out = self.drop(self.proj(out))
        return out, present_key_value

if __name__ == "__main__":
    
    torch.manual_seed(0)
    B, T, d = 2, 32, 48
    att = BudgetedCausalAttention(d, n_heads=4, span_min=2, temp=1.0)
    x = torch.randn(B, T, d)
    lo = ControllerSignal(g=torch.zeros(B, T), c=torch.zeros(B, T),
                          logits=torch.zeros(B, T, 2))
    hi = ControllerSignal(g=torch.ones(B, T), c=torch.ones(B, T),
                          logits=torch.zeros(B, T, 2))
    out_lo = att(x, lo)[0]
    out_hi = att(x, hi)[0]
    
    
    diff = (out_lo - out_hi).abs().mean().item()
    print(f"attn_budget smoke OK | out {tuple(out_hi.shape)} | "
          f"lo-vs-hi span mean|delta|={diff:.4f} (should be > 0)")
    assert diff > 1e-4, "span budget had no effect"
