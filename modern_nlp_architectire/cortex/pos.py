"""Positional handling — subsystem #2 (NoPE for length generalization).

Decoder-only transformers can generalize to unseen sequence lengths BETTER
without explicit positional encodings ("NoPE"): the causal mask already breaks
permutation symmetry, so the model infers position implicitly and is not tied to
a learned table that runs out past `max_seq_len`. That is the whole length-gen
story (train at L, test 2L–4L; see eval/length_gen.py).

Two schemes, config-selected via `model.pos` / `subsystems.nope`:
    'learned' : classic learned absolute position embedding (Stage 1 baseline).
    'nope'    : add nothing; positions emerge from causal structure.

The controller's per-token adaptive-depth use of the signal (also #2) is handled
by halting.py (#5, Stage 5); this file owns the positional half.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedPositional(nn.Module):
    """Absolute learned position embedding. forward(T)->(T, d)."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, T: int, device: torch.device, start_pos: int = 0) -> torch.Tensor:
        if start_pos + T > self.max_seq_len:
            raise ValueError(
                f"learned positions support T<= {self.max_seq_len}; got {start_pos + T}. "
                f"Use pos='nope' for length generalization."
            )
        return self.emb(torch.arange(start_pos, start_pos + T, device=device))          


class NoPE(nn.Module):
    """No positional encoding. forward(T)->0.0 (broadcasts; length-agnostic)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, T: int, device: torch.device, **kwargs) -> torch.Tensor:
        return torch.zeros((), device=device)                    


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # xq: (B, T, n_heads, d_head) -> (B, T, n_heads, d_head/2, 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # freqs_cis is (T, d_head/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (1, T, 1, d_head/2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 2048, rope_theta: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        # Precompute freqs_cis for max_seq_len.
        freqs_cis = precompute_freqs_cis(d_head, max_seq_len, theta=rope_theta)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, T: int, device: torch.device, start_pos: int = 0) -> torch.Tensor:
        # Return the slice of freqs_cis for the current sequence
        return self.freqs_cis[start_pos : start_pos + T].to(device)

def build_positional(mode: str, max_seq_len: int, d_model: int, d_head: int = 64, rope_theta: float = 10000.0) -> nn.Module:
    """Factory: 'learned' | 'nope' | 'rope' -> positional module."""
    if mode == "learned":
        return LearnedPositional(max_seq_len, d_model)
    if mode == "nope":
        return NoPE(d_model)
    if mode == "rope":
        return RotaryEmbedding(d_head, max_seq_len, rope_theta=rope_theta)
    raise ValueError(f"unknown positional mode: {mode}")


if __name__ == "__main__":
    
    d = 32
    nope = build_positional("nope", 128, d)
    learned = build_positional("learned", 128, d)
    dev = torch.device("cpu")
    assert nope(9999, dev).shape == ()                      
    assert learned(64, dev).shape == (64, d)
    try:
        learned(999, dev)
        raise AssertionError("learned should reject T>max")
    except ValueError:
        pass
    print("pos smoke OK | NoPE length-agnostic; learned bounded by max_seq_len")
