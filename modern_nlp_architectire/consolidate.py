from __future__ import annotations
from typing import Callable, Dict
import torch
import torch.nn.functional as F

def compute_importance(model, make_fn, spec, device, n_batches: int=16, batch_size: int=24, g_weighted: bool=True, seed: int=321) -> Dict[str, torch.Tensor]:
    imp = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    gen = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.train()
    for _ in range(n_batches):
        b = make_fn(batch_size, spec, gen, device=device)
        model.zero_grad(set_to_none=True)
        logits = model(b.x, gt_difficulty=b.difficulty)
        mask = b.difficulty > 0
        nll = F.cross_entropy(logits[mask], b.y[mask], reduction='none')
        if g_weighted and model.controller is not None:
            w = model.last_signal.g[mask].detach()
            L = (w * nll).sum() / w.sum().clamp_min(1e-06)
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

    def __init__(self, model, importance: Dict[str, torch.Tensor], lam: float=10000.0):
        self.lam = lam
        self.importance = importance
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model) -> torch.Tensor:
        loss = torch.zeros((), device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.importance:
                loss = loss + (self.importance[n] * (p - self.anchor[n]) ** 2).sum()
        return 0.5 * self.lam * loss
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cortex.model import build_model
    from data.synthetic import NeedleSpec, make_batch
    torch.manual_seed(0)
    base = dict(vocab_size=260, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_len=256)
    model = build_model(base, enable_controller=True)
    spec = NeedleSpec(seq_len=64)
    imp = compute_importance(model, make_batch, spec, torch.device('cpu'), n_batches=3, batch_size=8)
    cons = Consolidator(model, imp, lam=1000.0)
    p0 = cons.penalty(model).item()
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    p1 = cons.penalty(model).item()
    tot = sum((float(v.sum()) for v in imp.values()))
    print(f'consolidate smoke OK | Fisher sum={tot:.4f} (>=0) | penalty@anchor={p0:.4f} penalty@moved={p1:.4f}')
    assert tot >= 0 and p0 == 0.0 and (p1 > 0.0)