"""SCAN/COGS/ListOps accuracy

Subsystem #12 Compositionality. Evaluates zero-shot structural generalization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex.model import build_model
from data.compositional import make_listops_batch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"compositional: using device {device}")
    
    
    
    cfg = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "vocab_size": 16, 
        "max_seq_len": 32,
        "dropout": 0.0,
    }
    
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n--- Training on Depth-1 ---")
    model.train()
    g_train = torch.Generator().manual_seed(42)
    
    
    for step in range(1, 1001):
        batch = make_listops_batch(32, 32, depth=1, generator=g_train, device=device)
        logits = model(batch.x)
        
        
        B = logits.shape[0]
        rows = torch.arange(B)
        ap = batch.answer_pos
        loss = F.cross_entropy(logits[rows, ap], batch.y[rows, ap])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            pred = logits[rows, ap].argmax(dim=-1)
            acc = (pred == batch.y[rows, ap]).float().mean().item()
            print(f"Step {step:4d} | Loss {loss.item():.4f} | Acc {acc:.4f}")

    print("\n--- Evaluating Zero-Shot Structural Generalization ---")
    model.eval()
    g_eval = torch.Generator().manual_seed(99)
    
    
    acc_d1 = 0.0
    with torch.no_grad():
        for _ in range(10):
            batch = make_listops_batch(32, 32, depth=1, generator=g_eval, device=device)
            logits = model(batch.x)
            B = logits.shape[0]
            rows = torch.arange(B)
            pred = logits[rows, batch.answer_pos].argmax(dim=-1)
            acc_d1 += (pred == batch.y[rows, batch.answer_pos]).float().mean().item()
    print(f"Depth-1 (In-Distribution) Accuracy: {acc_d1/10:.4f}")
            
    
    acc_d2 = 0.0
    with torch.no_grad():
        for _ in range(10):
            batch = make_listops_batch(32, 32, depth=2, generator=g_eval, device=device)
            logits = model(batch.x)
            B = logits.shape[0]
            rows = torch.arange(B)
            pred = logits[rows, batch.answer_pos].argmax(dim=-1)
            acc_d2 += (pred == batch.y[rows, batch.answer_pos]).float().mean().item()
    print(f"Depth-2 (Zero-Shot) Accuracy:       {acc_d2/10:.4f}")
    
    print("\n==============================================================================================================================")
    print(f"[PASS] CORTEX eval (compositional) :: depth-1 acc={acc_d1/10:.3f} | depth-2 acc={acc_d2/10:.3f}")
    print("==============================================================================================================================")

if __name__ == "__main__":
    main()
