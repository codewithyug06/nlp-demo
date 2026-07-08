"""FLOP accounting — the denominator for every compute-vs-quality frontier.

Analytic (not profiled) FLOP counts for a dense decoder forward pass. This is
the baseline meter; adaptive subsystems in later stages report their *effective*
FLOPs as a fraction of this dense cost, which is the whole point of the thesis
(spend compute only where difficulty warrants).

Convention: one multiply-accumulate = 2 FLOPs. We count the dominant matmuls
(QKV, attention scores/values, output proj, MLP) and the attention score/context
products; we ignore norms, activations, and biases (negligible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FlopReport:
    per_token_forward: float   
    per_seq_forward: float     
    per_seq_train: float       
    breakdown: Dict[str, float]


def dense_flops(
    d_model: int,
    n_layers: int,
    d_ff: int,
    seq_len: int,
    vocab_size: int,
) -> FlopReport:
    """FLOPs for a dense decoder-only forward pass over one sequence of length T."""
    T = seq_len
    
    qkv = 2 * T * d_model * (3 * d_model)          
    attn_scores = 2 * T * T * d_model              
    attn_context = 2 * T * T * d_model             
    out_proj = 2 * T * d_model * d_model           
    mlp = 2 * T * d_model * d_ff * 2               
    per_layer = qkv + attn_scores + attn_context + out_proj + mlp

    layers_total = n_layers * per_layer
    lm_head = 2 * T * d_model * vocab_size
    per_seq_forward = layers_total + lm_head

    breakdown = {
        "attn_qkv": n_layers * qkv,
        "attn_scores": n_layers * attn_scores,
        "attn_context": n_layers * attn_context,
        "attn_out_proj": n_layers * out_proj,
        "mlp": n_layers * mlp,
        "lm_head": lm_head,
    }
    return FlopReport(
        per_token_forward=per_seq_forward / T,
        per_seq_forward=per_seq_forward,
        per_seq_train=3.0 * per_seq_forward,
        breakdown=breakdown,
    )


def dense_flops_from_cfg(model_cfg: Dict, seq_len: int) -> FlopReport:
    return dense_flops(
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        seq_len=seq_len,
        vocab_size=model_cfg["vocab_size"],
    )


if __name__ == "__main__":
    r = dense_flops(d_model=384, n_layers=6, d_ff=1536, seq_len=256, vocab_size=260)
    print(f"flops smoke OK | fwd/seq={r.per_seq_forward/1e9:.3f} GFLOP | "
          f"fwd/token={r.per_token_forward/1e6:.2f} MFLOP | "
          f"train/seq={r.per_seq_train/1e9:.3f} GFLOP")
    for k, v in r.breakdown.items():
        print(f"    {k:16s} {v/1e9:.3f} GFLOP")
