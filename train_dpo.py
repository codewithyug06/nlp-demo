import argparse
import copy
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from data.preference import make_preference_batch, NeedleSpec
from cortex.configuration_cortex import CortexConfig
from cortex.modeling_cortex import CortexForCausalLM
from cortex.losses import dpo_loss, controller_dpo_penalty
from utils import banner, get_device, get_logger, load_config, set_seed

log = get_logger("train_dpo")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cortex_micro.yaml")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--penalty_weight", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    
    
    spec = NeedleSpec(vocab_size=mcfg["vocab_size"], n_values=32, seq_len=dcfg["seq_len"])

    cortex_cfg = {
        "model": {
            "pos": mcfg.get("pos", "learned"),
            "span_min": mcfg.get("span_min", 4),
            "span_temp": mcfg.get("span_temp", 1.0),
            "residual": mcfg.get("residual", "vanilla"),
            "moe_experts": mcfg.get("moe_experts", 4),
            "moe_k_max": mcfg.get("moe_k_max", 2),
            "ponder_max_steps": mcfg.get("ponder_max_steps", 4),
            "ponder_min_steps": mcfg.get("ponder_min_steps", 1),
            "mem_slots": mcfg.get("mem_slots", 32),
        },
        "subsystems": {
            "controller": True,
            "dyn_patch": True,
            "attn_budget": True,
            "latent_loop": True,
            "moe": False,
            "memory": False,
        },
        "patch": {
            "patch_hidden": mcfg.get("patch_hidden", 32),
            "patch_ctx_weight": mcfg.get("patch_ctx_weight", 0.5),
        },
        "objective": {
            "mtp": False,
        }
    }
    
    hf_config = CortexConfig(
        vocab_size=mcfg["vocab_size"],
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        d_ff=mcfg["d_ff"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=mcfg.get("dropout", 0.0),
        cortex_cfg=cortex_cfg
    )

    log.info("Initializing models...")
    policy_model = CortexForCausalLM(hf_config).to(device)
    
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    optimizer = AdamW(policy_model.parameters(), lr=1e-4)
    
    max_steps = 50
    log.info(f"Starting DPO training for {max_steps} steps on synthetic preference pairs.")
    
    start_time = time.perf_counter()
    
    for step in range(1, max_steps + 1):
        batch = make_preference_batch(dcfg["batch_size"], spec, device=device)
        
        policy_model.train()
        
        
        pi_chosen_logits = policy_model(batch.chosen_x)
        g_chosen = policy_model.last_signal.g if policy_model.controller else None
        
        pi_rejected_logits = policy_model(batch.rejected_x)
        g_rejected = policy_model.last_signal.g if policy_model.controller else None
        
        
        with torch.no_grad():
            ref_chosen_logits = ref_model(batch.chosen_x)
            ref_rejected_logits = ref_model(batch.rejected_x)
            
        from cortex.losses import get_batch_logps
        
        pi_chosen_logps = get_batch_logps(pi_chosen_logits, batch.chosen_y, batch.chosen_difficulty)
        pi_rejected_logps = get_batch_logps(pi_rejected_logits, batch.rejected_y, batch.rejected_difficulty)
        
        ref_chosen_logps = get_batch_logps(ref_chosen_logits, batch.chosen_y, batch.chosen_difficulty)
        ref_rejected_logps = get_batch_logps(ref_rejected_logits, batch.rejected_y, batch.rejected_difficulty)
        
        loss_dpo = dpo_loss(pi_chosen_logps, pi_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=args.beta)
        
        loss_penalty = torch.tensor(0.0, device=device)
        if g_chosen is not None and g_rejected is not None:
            loss_penalty = args.penalty_weight * controller_dpo_penalty(g_chosen, g_rejected)
            
        total_loss = loss_dpo + loss_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            gc = g_chosen.mean().item() if g_chosen is not None else 0.0
            gr = g_rejected.mean().item() if g_rejected is not None else 0.0
            log.info(f"Step {step:3d} | DPO Loss: {loss_dpo.item():.4f} | Controller Pen: {loss_penalty.item():.4f} | g_chosen: {gc:.3f} | g_rejected: {gr:.3f}")
            
    wall = time.perf_counter() - start_time
    banner(True, "CORTEX DPO Run", f"100 steps in {wall:.1f}s")
    
if __name__ == "__main__":
    main()
