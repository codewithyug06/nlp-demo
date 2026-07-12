"""Signal-gated consolidation — subsystem #11 (resist forgetting).

EWC-style elastic consolidation with the CORTEX twist: the per-parameter
importance (a diagonal Fisher estimate) is weighted by g_t, so parameters that
serve HARD tokens (high difficulty/uncertainty) are protected most when a later
task is learned. After task A we snapshot θ*_A and importance Ω; while training
task B we add
    L_ewc = (λ/2) · Σ_θ Ω_θ (θ − θ*_A)²
which anchors important weights (incl. the KV-memory value slots, which are
parameters) near their task-A values.

Ω_θ = E_x[ Σ_t w_t (∂ NLL_t/∂θ)² ],   w_t = g_t (signal-gated) or 1 (plain EWC).
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F


def compute_importance(model, make_fn, spec, device, n_batches: int = 16,
                       batch_size: int = 24, g_weighted: bool = True,
                       seed: int = 321) -> Dict[str, torch.Tensor]:
    """g-weighted diagonal Fisher over trainable params. -> {name: Ω tensor}."""
    imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()
           if p.requires_grad}
    gen = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.train()
    for _ in range(n_batches):
        b = make_fn(batch_size, spec, gen, device=device)
        model.zero_grad(set_to_none=True)
        logits = model(b.x, gt_difficulty=b.difficulty)
        mask = b.difficulty > 0
        nll = F.cross_entropy(logits[mask], b.y[mask], reduction="none")   
        if g_weighted and model.controller is not None:
            w = model.last_signal.g[mask].detach()
            L = (w * nll).sum() / w.sum().clamp_min(1e-6)
        else:
            L = nll.mean()
        L.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                imp[n] += p.grad.detach() ** 2
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    for n in imp:
        imp[n] /= n_batches
    return imp


class Consolidator:
    """Holds task-A importance + anchor; yields the EWC penalty during task B."""

    def __init__(self, model, importance: Dict[str, torch.Tensor], lam: float = 1e4):
        self.lam = lam
        self.importance = importance
        self.anchor = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model) -> torch.Tensor:
        loss = torch.zeros((), device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.importance:
                loss = loss + (self.importance[n] * (p - self.anchor[n]) ** 2).sum()
        return 0.5 * self.lam * loss


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
    imp = compute_importance(model, make_batch, spec, torch.device("cpu"),
                             n_batches=3, batch_size=8)
    cons = Consolidator(model, imp, lam=1e3)
    p0 = cons.penalty(model).item()                       
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    p1 = cons.penalty(model).item()                       
    tot = sum(float(v.sum()) for v in imp.values())
    print(f"consolidate smoke OK | Fisher sum={tot:.4f} (>=0) | "
          f"penalty@anchor={p0:.4f} penalty@moved={p1:.4f}")
    assert tot >= 0 and p0 == 0.0 and p1 > 0.0
