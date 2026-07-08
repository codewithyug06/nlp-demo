"""Calibration — is g_t a calibrated uncertainty readout? (eval #10).

After the Stage-7 calibration loss, g_t should track the true per-token error
probability. We measure, over scored tokens:
  corr(g, NLL) : Pearson correlation of g_t with detached NLL (>0 = tracks difficulty)
  ece          : expected calibration error — |mean g in bin − empirical error rate|
                 averaged over confidence bins (0 = perfectly calibrated)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def calibration_metrics(model, make_fn, spec, batch_size: int, device,
                        n_batches: int = 16, n_bins: int = 10,
                        seed: int = 9090) -> Dict[str, float]:
    """corr(g,NLL) and ECE of g_t as an error-probability estimate on scored tokens."""
    assert model.controller is not None
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    gs, nlls, errs = [], [], []
    for _ in range(n_batches):
        b = make_fn(batch_size, spec, gen, device=device)
        logits = model(b.x)
        mask = b.difficulty > 0
        g = model.last_signal.g[mask]                      
        nll = F.cross_entropy(logits[mask], b.y[mask], reduction="none")
        pred = logits[mask].argmax(-1)
        err = (pred != b.y[mask]).float()                  
        gs.append(g); nlls.append(nll); errs.append(err)
    model.train()

    g = torch.cat(gs); nll = torch.cat(nlls); err = torch.cat(errs)
    
    gc, nc = g - g.mean(), nll - nll.mean()
    denom = (gc.norm() * nc.norm()).clamp_min(1e-8)
    corr = (gc * nc).sum() / denom

    
    ece = torch.zeros((), device=g.device)
    edges = torch.linspace(0, 1, n_bins + 1, device=g.device)
    for i in range(n_bins):
        m = (g >= edges[i]) & (g < edges[i + 1] if i < n_bins - 1 else g <= edges[i + 1])
        if m.any():
            ece += m.float().mean() * (g[m].mean() - err[m].mean()).abs()
    return {"corr_g_nll": corr.item(), "ece": ece.item(),
            "mean_g": g.mean().item(), "mean_err": err.mean().item()}


if __name__ == "__main__":
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cortex.model import build_model
    from data.synthetic import NeedleSpec, make_batch
    torch.manual_seed(0)
    base = dict(vocab_size=260, d_model=128, n_layers=2, n_heads=4, d_ff=256,
                max_seq_len=256)
    model = build_model(base, enable_controller=True)
    spec = NeedleSpec(seq_len=64)
    r = calibration_metrics(model, make_batch, spec, 8, torch.device("cpu"), n_batches=2)
    print("calibration smoke OK |", {k: round(v, 3) for k, v in r.items()})
