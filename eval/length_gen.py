"""Length generalization harness — subsystem #2 (train len L, test 2L–4L).

The needle task is a clean length-gen probe: retrieval must work no matter how
far the needle sits from the query. A model that memorized absolute positions
(learned pos) collapses past its trained length; a NoPE model should degrade
more gracefully. This harness just RUNS the sweep and reports accuracy per
length — Stage 3 acceptance is that it runs; the learned-vs-NoPE comparison is
carried forward to Stage 10.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from data.synthetic import NeedleSpec, make_batch, retrieval_accuracy


@torch.no_grad()
def length_gen_eval(model, vocab_size: int, n_values: int, lengths: List[int],
                    batch_size: int, device, n_batches: int = 8,
                    seed: int = 777) -> Dict[int, float]:
    """Retrieval accuracy at each sequence length. Skips lengths a learned-pos
    model cannot represent (returns NaN there rather than crashing)."""
    model.eval()
    max_len = getattr(model.pos, "max_seq_len", None)
    results: Dict[int, float] = {}
    for L in lengths:
        if max_len is not None and L > max_len:
            results[L] = float("nan")            
            continue
        spec = NeedleSpec(vocab_size=vocab_size, n_values=n_values, seq_len=L)
        g = torch.Generator().manual_seed(seed + L)
        accs = []
        for _ in range(n_batches):
            b = make_batch(batch_size, spec, g, device=device)
            accs.append(retrieval_accuracy(model(b.x), b))
        results[L] = sum(accs) / len(accs)
    model.train()
    return results


if __name__ == "__main__":
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cortex.model import build_model
    torch.manual_seed(0)
    base = dict(vocab_size=260, d_model=128, n_layers=2, n_heads=4, d_ff=256,
                max_seq_len=512)
    model = build_model(base, pos="nope")
    res = length_gen_eval(model, 260, 32, [64, 128, 256], batch_size=8,
                          device=torch.device("cpu"), n_batches=2)
    print("length_gen smoke OK |", {L: round(a, 3) for L, a in res.items()})
