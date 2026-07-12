"""
Distributed Training Scaffolding for CORTEX (Phase 2).

This script demonstrates how to set up `torch.distributed` and `FSDP` 
(Fully Sharded Data Parallel) for training large-scale CORTEX models.
It includes the logic to properly synchronize the custom MoE load-balancing
loss across GPU boundaries.

Note: Windows does not natively support NCCL backend required for multi-GPU
distributed training with PyTorch. This file is architectural scaffolding.
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

from modern_nlp_architectire.modeling import CortexForCausalLM
from modern_nlp_architectire.configuration import CortexConfig
from modern_nlp_architectire.block import TransformerBlock

def setup_distributed():
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this environment.")
    
    
    
    try:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank
    except Exception as e:
        print("Distributed setup failed (expected on Windows/non-NCCL environments).")
        print(f"Error: {e}")
        return -1

def get_fsdp_wrapped_model(model: nn.Module, device_id: int):
    
    cortex_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=cortex_auto_wrap_policy,
        device_id=device_id,
        
    )
    return model

def sync_moe_loss(local_aux_loss: torch.Tensor) -> torch.Tensor:
    """
    Synchronizes the MoE load-balancing loss across all GPUs.
    The local loss is summed across ranks and averaged.
    """
    if not dist.is_initialized():
        return local_aux_loss
        
    world_size = dist.get_world_size()
    
    dist.all_reduce(local_aux_loss, op=dist.ReduceOp.SUM)
    return local_aux_loss / world_size

def train_step(model, inputs, targets, optimizer):
    """
    Example step showing how custom MoE loss interacts with distributed training.
    """
    
    logits = model(inputs)
    
    
    ce_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    
    if hasattr(model, "last_moe_aux") and model.last_moe_aux is not None:
        
        moe_loss = sync_moe_loss(model.last_moe_aux)
    else:
        moe_loss = 0.0
        
    total_loss = ce_loss + 0.01 * moe_loss
    
    
    total_loss.backward()
    
    
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss

if __name__ == "__main__":
    print("--- CORTEX Distributed Training Scaffold ---")
    local_rank = setup_distributed()
    
    if local_rank != -1:
        
        config = CortexConfig(vocab_size=32000, d_model=1024, n_heads=16, d_ff=4096, num_layers=12, gradient_checkpointing=True)
        model = CortexForCausalLM(config).cuda()
        model = get_fsdp_wrapped_model(model, local_rank)
        print("Successfully initialized FSDP Cortex model.")
    else:
        print("Exiting scaffold execution.")
