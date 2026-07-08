"""Structured residual + norm placement — subsystem #7.

Two knobs the block consumes, both config-selected (this is a *study*: they must
run correctly and be measurable; they need not each help):

  norm placement (`norm`): 'pre' | 'post'  (classic pre-LN vs post-LN).

  residual scaling (`residual`):
     'vanilla'    : x + sublayer(x)
     'layerscale' : x + diag(γ) · sublayer(x), γ a learned per-channel scale
                    (LayerScale / ReZero-style), γ init small so blocks start
                    near-identity and depth is added gently.

`residual` also exposes an optional g_t coupling: when the controller marks a
token hard, its residual can be scaled up so hard tokens accrue more of each
sublayer's update. Kept OFF by default here (studied via config in Stage 10).
Shapes: (B, T, d).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """Per-channel learned residual scale (#7). out = γ ⊙ sublayer_out."""

    def __init__(self, d_model: int, init: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((d_model,), init))

    def forward(self, sub_out: torch.Tensor) -> torch.Tensor:
        return self.gamma * sub_out                       # (B,T,d) broadcast


class ResidualScaler(nn.Module):
    """Applies the structured-residual policy to a sublayer output before add."""

    def __init__(self, d_model: int, mode: str = "vanilla", init: float = 1e-2):
        super().__init__()
        self.mode = mode
        self.layerscale = LayerScale(d_model, init) if mode == "layerscale" else None

    def forward(self, sub_out: torch.Tensor) -> torch.Tensor:
        if self.mode == "vanilla":
            return sub_out
        if self.mode == "layerscale":
            return self.layerscale(sub_out)
        raise ValueError(f"unknown residual mode: {self.mode}")


if __name__ == "__main__":
    # Smoke test: layerscale starts near-identity-small; vanilla is a no-op.
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    van = ResidualScaler(16, "vanilla")
    ls = ResidualScaler(16, "layerscale", init=1e-2)
    assert torch.equal(van(x), x)
    assert torch.allclose(ls(x), 1e-2 * x)
    print("residual smoke OK | vanilla no-op; layerscale gamma*out (init 1e-2)")
