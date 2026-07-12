"""CORTEX ablation driver (§2 rule 5: every ablation is config-only).
"""

from __future__ import annotations

import argparse
import yaml
import torch
import torch.nn.functional as F
import os
import sys

from modern_nlp_architectire.model import build_model
from data.synthetic import make_batch as make_needle_batch, NeedleSpec

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_ablation(run_name: str, overrides: dict, device: str) -> float:
    
    cfg = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "vocab_size": 260,
        "max_seq_len": 64,
        "dropout": 0.0,
    }
    
    
    features = {
        "enable_controller": overrides.get("enable_controller", True),
        "pos": overrides.get("pos", "nope"),
        "dyn_patch": overrides.get("dyn_patch", False),
        "attn_budget": overrides.get("attn_budget", False),
        "residual": overrides.get("residual", "vanilla"),
        "latent_ponder": overrides.get("latent_ponder", False),
        "moe": overrides.get("moe", False),
        "memory": overrides.get("memory", False),
        "mtp": overrides.get("mtp", False),
    }
    
    model = build_model(cfg, **features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    model.train()
    g = torch.Generator().manual_seed(42)
    spec = NeedleSpec(vocab_size=260, n_values=32, seq_len=64)
    loss_sum = 0.0
    for step in range(20):
        b = make_needle_batch(16, spec, g, device=device)
        logits = model(b.x, gt_difficulty=b.difficulty)
        
        
        B, T = b.x.shape
        flat_logits = logits.view(B*T, -1)
        flat_y = b.y.view(B*T)
        loss = F.cross_entropy(flat_logits, flat_y, ignore_index=0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        
    return loss_sum / 20.0

def main() -> None:
    print("Running Micro-Scale Ablation Grid...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    grid = []
    
    
    grid.append(("Dense Baseline", {"enable_controller": False}))
    
    
    coupled_feats = {
        "enable_controller": True,
        "dyn_patch": True,
        "attn_budget": True,
        "residual": "layerscale",
        "latent_ponder": True,
        "moe": True,
        "memory": True,
        "mtp": True
    }
    grid.append(("CORTEX Coupled (Full)", coupled_feats.copy()))
    
    
    grid.append(("CORTEX Decoupled", coupled_feats.copy()))
    
    
    for k in ["dyn_patch", "attn_budget", "latent_ponder", "moe", "memory", "mtp"]:
        lo = coupled_feats.copy()
        lo[k] = False
        grid.append((f"LOO: w/o {k}", lo))
        
    
    os.makedirs("results", exist_ok=True)
    with open("results/table.md", "w") as f:
        f.write("# CORTEX Ablation Grid (Micro-Scale)\n\n")
        f.write("| Model Variant | Loss (20 steps) | Note |\n")
        f.write("|---------------|-----------------|------|\n")
        
        for name, feats in grid:
            print(f"Training {name}...")
            loss = run_ablation(name, feats, device)
            f.write(f"| {name} | {loss:.4f} | |\n")
            
    print("Ablation grid complete! Results saved to results/table.md")

if __name__ == "__main__":
    main()
