"""Mock preference dataset for Direct Preference Optimization (DPO).

Generates synthetic pairs of (chosen, rejected) continuations for the same prompt.
We use the Needle task format:
Prompt: BOS filler... NEEDLE V filler... QUERY
Chosen: V
Rejected: W (where W != V)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from data.synthetic import NeedleSpec, PAD, BOS, NEEDLE, QUERY

@dataclass
class PreferenceBatch:
    prompt_len: int                 
    chosen_x: torch.Tensor          
    chosen_y: torch.Tensor          
    chosen_difficulty: torch.Tensor 
    rejected_x: torch.Tensor        
    rejected_y: torch.Tensor        
    rejected_difficulty: torch.Tensor 


def make_preference_batch(
    batch_size: int,
    spec: NeedleSpec,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str = "cpu",
) -> PreferenceBatch:
    """Generate one reproducible batch of preference pairs. Shapes: (B, T)."""
    B, T = batch_size, spec.seq_len
    g = generator

    def randint(lo: int, hi: int, size) -> torch.Tensor:
        return torch.randint(lo, hi, size, generator=g)

    
    x = randint(spec.filler_lo, spec.filler_hi, (B, T))
    x[:, 0] = BOS

    
    values = randint(spec.sample_lo, spec.sample_hi, (B,))
    
    
    wrong_values = randint(spec.sample_lo, spec.sample_hi - 1, (B,))
    wrong_values = torch.where(wrong_values >= values, wrong_values + 1, wrong_values)

    x[:, T - 2] = QUERY
    
    
    needle_pos = randint(3, T - 4, (B,))
    rows = torch.arange(B)
    x[rows, needle_pos] = NEEDLE
    off = 1 if spec.direction == "after" else -1
    x[rows, needle_pos + off] = values

    
    chosen_x = x.clone()
    chosen_x[:, T - 1] = values
    chosen_y = torch.full((B, T), PAD, dtype=torch.long)
    chosen_y[:, : T - 1] = chosen_x[:, 1:]
    
    chosen_difficulty = torch.zeros((B, T), dtype=torch.float32)
    answer_pos = torch.full((B,), T - 2, dtype=torch.long)
    chosen_difficulty[rows, answer_pos] = 1.0

    
    rejected_x = x.clone()
    rejected_x[:, T - 1] = wrong_values
    rejected_y = torch.full((B, T), PAD, dtype=torch.long)
    rejected_y[:, : T - 1] = rejected_x[:, 1:]
    
    rejected_difficulty = torch.zeros((B, T), dtype=torch.float32)
    rejected_difficulty[rows, answer_pos] = 1.0

    return PreferenceBatch(
        prompt_len=T - 1,
        chosen_x=chosen_x.to(device),
        chosen_y=chosen_y.to(device),
        chosen_difficulty=chosen_difficulty.to(device),
        rejected_x=rejected_x.to(device),
        rejected_y=rejected_y.to(device),
        rejected_difficulty=rejected_difficulty.to(device),
    )

if __name__ == "__main__":
    spec = NeedleSpec(vocab_size=260, n_values=32, seq_len=16)
    g1 = torch.Generator().manual_seed(0)
    batch = make_preference_batch(4, spec, g1)
    
    
    assert torch.equal(batch.chosen_x[:, :batch.prompt_len], batch.rejected_x[:, :batch.prompt_len])
    
    assert not torch.equal(batch.chosen_x[:, batch.prompt_len], batch.rejected_x[:, batch.prompt_len])
    print("preference batch smoke OK")
