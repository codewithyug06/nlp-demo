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

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        if T > self.max_seq_len:
            raise ValueError(
                f"learned positions support T<= {self.max_seq_len}; got {T}. "
                f"Use pos='nope' for length generalization."
            )
        return self.emb(torch.arange(T, device=device))          


class NoPE(nn.Module):
    """No positional encoding. forward(T)->0.0 (broadcasts; length-agnostic)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device)                    


def build_positional(mode: str, max_seq_len: int, d_model: int) -> nn.Module:
    """Factory: 'learned' | 'nope' -> positional module."""
    if mode == "learned":
        return LearnedPositional(max_seq_len, d_model)
    if mode == "nope":
        return NoPE(d_model)
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
