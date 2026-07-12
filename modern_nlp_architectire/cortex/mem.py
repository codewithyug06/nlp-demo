"""Uncertainty-gated KV memory — subsystem #6.

A small bank of M learnable memory slots (keys + values). Each token READS from
the bank by attention, but the read is GATED by g_t (difficulty/uncertainty):
    h'_t = h_t + g_t · W_r · Σ_m softmax(h_t·K_m) V_m
so memory only fires on hard/uncertain tokens (the acceptance criterion). A
g-gated WRITE accumulates high-g token states into a persistent EMA buffer added
to the bank on later steps — a genuine read/write memory that adapts across steps
without within-sequence causal leakage (writes affect only subsequent forwards).

Shapes: (B, T, d).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import ControllerSignal


@dataclass
class MemInfo:
    read_gate_mean: float        
    write_mass: float            
    slot_usage: torch.Tensor     


class GatedKVMemory(nn.Module):
    """g_t-gated read + g_t-gated cross-step write into a KV bank (#6)."""

    def __init__(self, d_model: int, *, n_slots: int = 32, write_ema: float = 0.99):
        super().__init__()
        self.n_slots = n_slots
        self.scale = d_model ** -0.5
        self.keys = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.read_proj = nn.Linear(d_model, d_model)
        self.write_ema = write_ema
        
        self.register_buffer("write_buf", torch.zeros(n_slots, d_model))
        self.last_info: Optional[MemInfo] = None

    def forward(self, h: torch.Tensor,
                signal: Optional[ControllerSignal] = None) -> torch.Tensor:
        B, T, d = h.shape
        g = (signal.g if signal is not None
             else torch.ones(B, T, device=h.device, dtype=h.dtype))   

        vals = self.values + self.write_buf                          
        att = F.softmax((h @ self.keys.t()) * self.scale, dim=-1)     
        read = att @ vals                                            
        h_out = h + g.unsqueeze(-1) * self.read_proj(read)           

        
        if self.training:
            with torch.no_grad():
                wgate = g.reshape(-1, 1)                             
                hn = h.reshape(-1, d)                                
                wa = F.softmax((hn @ self.keys.t()) * self.scale, dim=-1)  
                
                upd = (wa * wgate).t() @ hn                          
                norm = (wa * wgate).sum(0).clamp_min(1e-6).unsqueeze(-1)
                upd = upd / norm
                self.write_buf.mul_(self.write_ema).add_(
                    (1 - self.write_ema) * upd)
                write_mass = wgate.mean().item()
        else:
            write_mass = 0.0

        self.last_info = MemInfo(read_gate_mean=g.mean().item(),
                                 write_mass=write_mass,
                                 slot_usage=att.mean(dim=(0, 1)).detach())
        return h_out


if __name__ == "__main__":
    
    torch.manual_seed(0)
    B, T, d = 2, 16, 48
    mem = GatedKVMemory(d, n_slots=16)
    mem.eval()
    h = torch.randn(B, T, d)
    z = ControllerSignal(g=torch.zeros(B, T), c=torch.zeros(B, T),
                         logits=torch.zeros(B, T, 2))
    o = ControllerSignal(g=torch.ones(B, T), c=torch.ones(B, T),
                         logits=torch.zeros(B, T, 2))
    out0, out1 = mem(h, z), mem(h, o)
    d0 = (out0 - h).abs().mean().item()
    d1 = (out1 - h).abs().mean().item()
    print(f"mem smoke OK | |read| at g=0 -> {d0:.4f} (should be 0) | "
          f"at g=1 -> {d1:.4f} (should be > 0)")
    assert d0 < 1e-7 < d1, "memory read is not g-gated"
