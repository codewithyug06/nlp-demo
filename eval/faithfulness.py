"""Faithfulness — does g_t fire on ground-truth hard tokens? (eval #1/#10).

The make-or-break test for the whole thesis (Stage 4). We do NOT supervise g_t
on the difficulty labels; we only let the attention-budget mechanism + a compute
cost shape it, then ask: on coreference, does the controller's g_t spike on the
ANAPHOR (the one token whose ground-truth difficulty = 1) relative to easy
tokens?

Reports:
  g_hard_mean, g_easy_mean      : mean g_t at anaphor vs elsewhere
  separation = hard - easy      : primary faithfulness signal
  auroc                         : how well g_t ranks hard vs easy tokens (0.5=chance)
  passed                        : separation > threshold and auroc > 0.5+margin
"""

from __future__ import annotations

from typing import Dict

import torch

from data.synthetic import CorefSpec, make_coref_batch


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC via the rank statistic (labels in {0,1}). Chance = 0.5."""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float)
    r_pos = ranks[labels > 0.5].sum()
    n_p, n_n = pos.numel(), neg.numel()
    return ((r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n)).item()


@torch.no_grad()
def faithfulness_coref(model, spec: CorefSpec, batch_size: int, device,
                       n_batches: int = 8, sep_threshold: float = 0.10,
                       seed: int = 4242) -> Dict[str, float]:
    """Measure whether g_t spikes on coref anaphors (ground-truth hard tokens)."""
    assert model.controller is not None, "faithfulness needs the controller (#10)"
    model.eval()
    g = torch.Generator().manual_seed(seed)
    g_all, lab_all = [], []
    for _ in range(n_batches):
        b = make_coref_batch(batch_size, spec, g, device=device)
        model(b.x)                                        
        gt = model.last_signal.g                          
        g_all.append(gt.reshape(-1))
        lab_all.append(b.difficulty.reshape(-1))
    model.train()

    gv = torch.cat(g_all)
    lb = torch.cat(lab_all)
    hard = gv[lb > 0.5]
    easy = gv[lb <= 0.5]
    g_hard = hard.mean().item()
    g_easy = easy.mean().item()
    sep = g_hard - g_easy
    auroc = _auroc(gv, lb)
    passed = (sep > sep_threshold) and (auroc > 0.5 + 0.05)
    return {"g_hard_mean": g_hard, "g_easy_mean": g_easy, "separation": sep,
            "auroc": auroc, "passed": bool(passed)}


if __name__ == "__main__":
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cortex.model import build_model
    torch.manual_seed(0)
    base = dict(vocab_size=260, d_model=128, n_layers=2, n_heads=4, d_ff=256,
                max_seq_len=256)
    model = build_model(base, enable_controller=True, attn_budget=True)
    spec = CorefSpec(seq_len=64, n_pairs=4)
    r = faithfulness_coref(model, spec, 8, torch.device("cpu"), n_batches=2)
    print("faithfulness smoke OK |", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in r.items()})
