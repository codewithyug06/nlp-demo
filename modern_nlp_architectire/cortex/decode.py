"""Difficulty-driven speculative decoding — subsystem #13.

Self-speculation: the model drafts cheaply (its own forward with the latent
ponder loop skipped) and verifies with a single full forward. g_t gates the
speculation window — the drafter stops proposing once it hits a HARD token
(g_t > threshold), so easy runs are accepted in parallel while hard tokens get
an individual full-model verification. Greedy output is reproduced EXACTLY
(standard speculative guarantee: accept a draft token only where it equals the
verifier's greedy token; on mismatch take the verifier's token), while the number
of sequential VERIFY forwards (the expensive step) drops.

Metrics: n_verify (sequential target calls) vs greedy's n; acceptance rate;
matches_greedy (must be True).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class DecodeStats:
    tokens: List[int]
    n_verify: int          
    n_draft: int           
    n_generated: int
    acceptance: float      
    matches_greedy: bool


@torch.no_grad()
def _last(model, seq: List[int], device, skip_ponder: bool):
    """Forward on `seq`; return (last-pos logits (V,), last-pos g scalar)."""
    x = torch.tensor(seq, device=device)[None, :]
    logits = model(x, skip_ponder=skip_ponder)
    g = (model.last_signal.g[0, -1].item() if model.controller is not None else 0.0)
    return logits[0, -1], g


@torch.no_grad()
def greedy_decode(model, prompt: List[int], n_new: int, device) -> DecodeStats:
    """Plain greedy: one full forward per token."""
    model.eval()
    seq = list(prompt)
    for _ in range(n_new):
        logits, _ = _last(model, seq, device, skip_ponder=False)
        seq.append(int(logits.argmax()))
    return DecodeStats(seq, n_verify=n_new, n_draft=0, n_generated=n_new,
                       acceptance=0.0, matches_greedy=True)


@torch.no_grad()
def speculative_decode(model, prompt: List[int], n_new: int, device,
                       gamma: int = 4, g_threshold: float = 0.6) -> DecodeStats:
    """Difficulty-gated self-speculative decoding. Reproduces greedy exactly."""
    model.eval()
    seq = list(prompt)
    target_len = len(prompt) + n_new
    n_verify = n_draft = drafted = accepted = 0

    while len(seq) < target_len:
        
        draft: List[int] = []
        cur = list(seq)
        for _ in range(gamma):
            logits, g = _last(model, cur, device, skip_ponder=True)
            tok = int(logits.argmax())
            draft.append(tok); cur.append(tok); n_draft += 1
            if g > g_threshold:            
                break
        drafted += len(draft)

        
        x = torch.tensor(seq + draft, device=device)[None, :]
        vlogits = model(x, skip_ponder=False)[0]                 
        n_verify += 1
        L0 = len(seq)

        
        n_acc = 0
        for i, dt in enumerate(draft):
            target_tok = int(vlogits[L0 + i - 1].argmax())
            if target_tok != dt:
                break
            n_acc += 1
        accepted += n_acc

        if n_acc == len(draft):
            seq.extend(draft)                                   
            if len(seq) < target_len:                          
                seq.append(int(vlogits[L0 + len(draft) - 1].argmax()))
        else:
            seq.extend(draft[:n_acc])                           
            seq.append(int(vlogits[L0 + n_acc - 1].argmax()))  

    seq = seq[:target_len]
    greedy = greedy_decode(model, prompt, n_new, device).tokens
    return DecodeStats(seq, n_verify=n_verify, n_draft=n_draft, n_generated=n_new,
                       acceptance=(accepted / max(1, drafted)),
                       matches_greedy=(seq == greedy))


if __name__ == "__main__":
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cortex.model import build_model
    torch.manual_seed(0)
    base = dict(vocab_size=260, d_model=128, n_layers=2, n_heads=4, d_ff=256,
                max_seq_len=256)
    model = build_model(base, enable_controller=True, latent_ponder=True)
    prompt = [1, 40, 41, 42]
    dev = torch.device("cpu")
    spec = speculative_decode(model, prompt, 24, dev, gamma=4)
    print(f"decode smoke OK | matches_greedy={spec.matches_greedy} | "
          f"verify={spec.n_verify} vs greedy={spec.n_generated} "
          f"(speedup {spec.n_generated/spec.n_verify:.2f}x) | "
          f"acceptance={spec.acceptance:.2f}")
    assert spec.matches_greedy, "speculative decode diverged from greedy!"
