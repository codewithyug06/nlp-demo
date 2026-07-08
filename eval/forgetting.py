"""Sequential-task retention — subsystem #11 (does consolidation resist forgetting?).

Two conflicting ASSOCIATIVE tasks: the same K keys map to DIFFERENT values under
mapping A vs mapping B. The model must MEMORISE the table (values never appear in
the input), so training B overwrites A — the canonical catastrophic-forgetting
setup. We train A, snapshot g-weighted importance (#11), train B, then re-measure
A. Consolidation should beat the no-consolidation baseline (reported honestly
either way, §2 r8).

Runnable: `python -m eval.forgetting`  (prints the A/B retention comparison).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from data.synthetic import BOS, N_SPECIAL, retrieval_accuracy
from cortex.consolidate import Consolidator, compute_importance
from cortex.losses import lm_loss
from cortex.model import build_model

N_KEYS = 8
N_VALUES = 32
VOCAB = 260
KEY_LO = N_SPECIAL                       
VAL_LO = N_SPECIAL + N_KEYS              
FILLER_LO = VAL_LO + N_VALUES
CHANCE = 1.0 / N_VALUES


@dataclass
class AssocBatch:
    x: torch.Tensor
    y: torch.Tensor
    difficulty: torch.Tensor
    answer_pos: torch.Tensor


def make_mapping(gen: torch.Generator) -> torch.Tensor:
    """Random key->value table. (K,) value tokens."""
    return VAL_LO + torch.randint(0, N_VALUES, (N_KEYS,), generator=gen)


def make_assoc(mapping: torch.Tensor, batch_size: int, seq_len: int,
               gen: torch.Generator, device) -> AssocBatch:
    """BOS filler... KEY VALUE; predict VALUE=mapping[key] (must be memorised)."""
    B, T = batch_size, seq_len
    x = torch.randint(FILLER_LO, VOCAB, (B, T), generator=gen)
    x[:, 0] = BOS
    keyidx = torch.randint(0, N_KEYS, (B,), generator=gen)
    x[:, T - 2] = KEY_LO + keyidx
    x[:, T - 1] = mapping[keyidx]
    y = torch.full((B, T), 0, dtype=torch.long)
    y[:, : T - 1] = x[:, 1:]
    diff = torch.zeros(B, T)
    ap = torch.full((B,), T - 2)
    diff[torch.arange(B), ap] = 1.0
    return AssocBatch(x.to(device), y.to(device), diff.to(device), ap.to(device))


@torch.no_grad()
def acc_on(model, mapping, seq_len, device, batch_size=64, n_batches=8,
           seed=555) -> float:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    accs = []
    for _ in range(n_batches):
        b = make_assoc(mapping, batch_size, seq_len, g, device)
        accs.append(retrieval_accuracy(model(b.x), b))
    model.train()
    return sum(accs) / len(accs)


def train_on(model, mapping, steps, opt, device, seq_len,
             consolidator: Optional[Consolidator] = None, batch_size=64,
             seed=1, imp_accum: Optional[dict] = None,
             imp_g_weighted: bool = True) -> None:
    """Train on one mapping. If imp_accum is given, accumulate g-weighted online
    importance (grad^2) THROUGHOUT training — non-zero even after convergence,
    unlike post-hoc Fisher on a memorized task."""
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        b = make_assoc(mapping, batch_size, seq_len, g, device)
        logits = model(b.x, gt_difficulty=b.difficulty)
        loss = lm_loss(logits, b)
        if consolidator is not None:
            loss = loss + consolidator.penalty(model)          
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if imp_accum is not None:                              
            model.zero_grad(set_to_none=True)
            lg = model(b.x, gt_difficulty=b.difficulty)
            mask = b.difficulty > 0
            nll = F.cross_entropy(lg[mask], b.y[mask], reduction="none")
            if imp_g_weighted and model.controller is not None:
                w = model.last_signal.g[mask].detach()
                imp_loss = (w * nll).sum() / w.sum().clamp_min(1e-6)
            else:
                imp_loss = nll.mean()
            imp_loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    imp_accum[n] += p.grad.detach() ** 2
            model.zero_grad(set_to_none=True)


def run_continual(consolidate: bool, lam: float = 1e7, g_weighted: bool = True,
                  steps: int = 300, seq_len: int = 24, seed: int = 0,
                  device: Optional[torch.device] = None) -> Dict[str, float]:
    """Train A -> (consolidate) -> train B; return A-retention and B-acc.

    Uses SGD (not Adam): Adam normalizes each param's step to ~lr, which cancels
    the EWC restoring force, so consolidation cannot bind. lambda is large to
    match the small online-importance magnitudes.
    """
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    base = dict(vocab_size=VOCAB, d_model=256, n_layers=4, n_heads=4, d_ff=1024,
                max_seq_len=64)
    model = build_model(base, enable_controller=True).to(device)
    mg = torch.Generator().manual_seed(seed + 7)
    mapping_A = make_mapping(mg)
    mapping_B = make_mapping(mg)                                
    opt = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9)

    
    imp = ({n: torch.zeros_like(p) for n, p in model.named_parameters()
            if p.requires_grad} if consolidate else None)
    train_on(model, mapping_A, steps, opt, device, seq_len, imp_accum=imp,
             imp_g_weighted=g_weighted)                         
    accA_before = acc_on(model, mapping_A, seq_len, device)

    cons = None
    if consolidate:
        for n in imp:
            imp[n] /= steps
        cons = Consolidator(model, imp, lam=lam)

    train_on(model, mapping_B, steps, opt, device, seq_len, consolidator=cons)
    accA_after = acc_on(model, mapping_A, seq_len, device)
    accB = acc_on(model, mapping_B, seq_len, device)
    return {"accA_before": accA_before, "accA_after": accA_after, "accB": accB,
            "forgetting": accA_before - accA_after, "retention": accA_after}


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = run_continual(consolidate=False, device=dev)
    cons = run_continual(consolidate=True, device=dev)
    print("=== #11 continual (task A=after, task B=before) ===")
    print(f"baseline (no consolidation): A_before={base['accA_before']:.3f} "
          f"A_after={base['accA_after']:.3f} B={base['accB']:.3f} "
          f"forgetting={base['forgetting']:.3f}")
    print(f"consolidated (g-weighted):   A_before={cons['accA_before']:.3f} "
          f"A_after={cons['accA_after']:.3f} B={cons['accB']:.3f} "
          f"forgetting={cons['forgetting']:.3f}")
    print(f"retention gain (A_after cons - base) = "
          f"{cons['accA_after'] - base['accA_after']:+.3f}")
