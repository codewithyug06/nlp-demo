"""Universal Compute Controller — subsystem #10 (THE core idea).

A tiny per-token module that reads the residual stream and emits ONE shared
difficulty signal `(g_t, c_t)`:

    g_t in (0,1): difficulty / compute gate. High => "this token is hard, spend
                  more compute here." Every downstream subsystem (#1..#9,#11,#13)
                  consumes this same g_t to set its per-token compute policy.
    c_t in (0,1): confidence. c_t = P(model is right here); (1 - c_t) is the
                  calibrated "I-don't-know" readout (#10 / uncertainty).

This file is ONLY the readout. It gains power over the stack in later stages;
in Stage 2 it is wired in INERT (computed, logged, consumed by nothing) to prove
the plumbing is neutral. Its calibration training arrives in Stage 7 (losses.py).

Equation: [g_logit, c_logit] = W2 · GELU(W1 · h_t);  g_t = σ(g_logit),
          c_t = σ(c_logit).   h_t: (B,T,d) -> signal: (B,T) each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ControllerSignal:
    """One shared per-token signal. g, c: (B, T). logits: (B, T, 2) (pre-sigmoid)."""
    g: torch.Tensor           # difficulty / compute gate in (0,1)
    c: torch.Tensor           # confidence in (0,1); uncertainty = 1 - c
    logits: torch.Tensor      # raw [g_logit, c_logit] for calibration losses later

    def detach(self) -> "ControllerSignal":
        return ControllerSignal(self.g.detach(), self.c.detach(), self.logits.detach())


class Controller(nn.Module):
    """Per-token difficulty/uncertainty head. (B,T,d) -> ControllerSignal."""

    def __init__(self, d_model: int, hidden: Optional[int] = None,
                 g_bias: float = 0.0):
        super().__init__()
        hidden = hidden or max(16, d_model // 4)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        # Bias the difficulty gate at init (e.g. slightly low) so the model can
        # start "cheap" and learn where to spend; 0.0 => centred at 0.5.
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.bias[0] = g_bias

    def forward(self, h: torch.Tensor) -> ControllerSignal:
        logits = self.fc2(F.gelu(self.fc1(h)))     # (B, T, 2)
        g = torch.sigmoid(logits[..., 0])           # (B, T)
        c = torch.sigmoid(logits[..., 1])           # (B, T)
        return ControllerSignal(g=g, c=c, logits=logits)


if __name__ == "__main__":
    # Smoke test: shapes + ranges.
    torch.manual_seed(0)
    ctrl = Controller(d_model=384)
    h = torch.randn(2, 16, 384)
    sig = ctrl.forward(h)
    assert sig.g.shape == (2, 16) and sig.c.shape == (2, 16)
    assert float(sig.g.min()) >= 0.0 and float(sig.g.max()) <= 1.0
    print(f"controller smoke OK | g mean={sig.g.mean():.3f} "
          f"c mean={sig.c.mean():.3f} | params={sum(p.numel() for p in ctrl.parameters())}")
