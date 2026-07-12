"""Synthetic mini-ListOps task for compositional generalization evaluation.

Tests if a model trained on depth-1 expressions (e.g. MAX(a, b)) can structurally
generalize zero-shot to depth-2+ expressions (e.g. MAX(MIN(a, b), c)).
"""

import torch
from dataclasses import dataclass

PAD, BOS, QUERY = 0, 1, 2
MAX_OP, MIN_OP = 3, 4
VAL_OFFSET = 5
N_VALS = 10

@dataclass
class ListOpsBatch:
    x: torch.Tensor
    y: torch.Tensor
    difficulty: torch.Tensor
    answer_pos: torch.Tensor

def generate_tree(depth, g):
    if depth == 0:
        val = int(torch.randint(0, N_VALS, (1,), generator=g).item())
        return [VAL_OFFSET + val], val
    
    op = int(torch.randint(0, 2, (1,), generator=g).item())
    op_tok = MAX_OP if op == 0 else MIN_OP
    
    left_depth = depth - 1
    right_depth = 0 if depth == 1 else int(torch.randint(0, depth, (1,), generator=g).item())
    
    if torch.rand(1, generator=g).item() > 0.5:
        left_depth, right_depth = right_depth, left_depth
        
    left_toks, left_val = generate_tree(left_depth, g)
    right_toks, right_val = generate_tree(right_depth, g)
    
    ans = max(left_val, right_val) if op == 0 else min(left_val, right_val)
    return [op_tok] + left_toks + right_toks, ans

def make_listops_batch(batch_size, seq_len, depth, generator=None, device="cpu"):
    B, T = batch_size, seq_len
    x = torch.full((B, T), PAD, dtype=torch.long)
    y = torch.full((B, T), PAD, dtype=torch.long)
    answer_pos = torch.zeros(B, dtype=torch.long)
    difficulty = torch.zeros((B, T), dtype=torch.float32)
    
    for b in range(B):
        toks, ans = generate_tree(depth, generator)
        x[b, 0] = BOS
        L = len(toks)
        
        
        if L + 2 > T:
            toks = toks[:T-2]
            L = len(toks)
            
        x[b, 1:L+1] = torch.tensor(toks, dtype=torch.long)
        x[b, L+1] = QUERY
        
        y[b, :L+1] = x[b, 1:L+2]
        y[b, L+1] = VAL_OFFSET + ans
        
        answer_pos[b] = L + 1
        difficulty[b, L+1] = 1.0
        
    return ListOpsBatch(
        x=x.to(device), 
        y=y.to(device), 
        difficulty=difficulty.to(device), 
        answer_pos=answer_pos.to(device)
    )
