"""Assemble the decoder-only transformer (dense baseline + input stack).

Stage 1: plain decoder (`enable_controller=False`, learned pos) — the baseline.
Stage 2: `enable_controller=True` adds the Universal Controller (#10), inert.
Stage 3: input stack — positional scheme (#2 learned|nope) and optional dynamic
         byte-patching (#3) that consumes the controller's g_t. A fixed-tokenizer
         fallback (`dyn_patch=False`) guarantees Stage-1 comparability.

Transformer internals live in `block.py`. Dims come from config (§6).
Shapes annotated as (B, T, d).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .block import TransformerBlock
from .controller import Controller, ControllerSignal
from .dyn_patch import DynamicPatcher, PatchOutput
from .halting import PonderInfo
from .latent_loop import LatentPonder
from .mem import GatedKVMemory
from .pos import build_positional


@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.0
    norm: str = "pre"                 
    tie_embeddings: bool = True
    enable_controller: bool = False   
    pos: str = "learned"              
    dyn_patch: bool = False           
    patch_hidden: int = 64
    patch_ctx_weight: float = 0.5
    patch_use_signal: bool = True     
    attn_budget: bool = False         
    span_min: int = 8
    span_temp: float = 4.0
    residual: str = "vanilla"         
    latent_ponder: bool = False       
    ponder_max_steps: int = 4
    ponder_min_steps: int = 1
    ponder_prior_lambda: float = 0.5
    ponder_g_bias: float = 2.0
    moe: bool = False                 
    moe_experts: int = 4
    moe_k_max: int = 2
    moe_expert_ff: int = 0            
    memory: bool = False              
    mem_slots: int = 32
    mtp: bool = False                 

    @classmethod
    def from_dict(cls, m: Dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in m.items() if k in fields})


class DenseDecoder(nn.Module):
    """Decoder-only LM. forward(x)->logits (B,T,V); stores last signal/patch info."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = build_positional(cfg.pos, cfg.max_seq_len, cfg.d_model)  
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.norm,
                             attn_budget=cfg.attn_budget, span_min=cfg.span_min,
                             span_temp=cfg.span_temp, residual=cfg.residual,
                             moe=cfg.moe, moe_experts=cfg.moe_experts,
                             moe_k_max=cfg.moe_k_max, moe_expert_ff=cfg.moe_expert_ff)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.mtp_head = (nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
                         if cfg.mtp else None)          
        self.apply(self._init_weights)

        
        
        self.controller: Optional[Controller] = None
        if cfg.enable_controller:
            self.controller = Controller(cfg.d_model)
            self.controller.apply(self._init_weights)
        self.patcher: Optional[DynamicPatcher] = None
        if cfg.dyn_patch:
            self.patcher = DynamicPatcher(cfg.d_model, cfg.patch_hidden,
                                          cfg.patch_ctx_weight, cfg.patch_use_signal)
            self.patcher.apply(self._init_weights)

        self.memory: Optional[GatedKVMemory] = None
        if cfg.memory:
            self.memory = GatedKVMemory(cfg.d_model, cfg.mem_slots)
            self.memory.apply(self._init_weights)

        self.ponder: Optional[LatentPonder] = None
        if cfg.latent_ponder:
            self.ponder = LatentPonder(
                cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.norm,
                cfg.attn_budget, cfg.span_min, cfg.span_temp, cfg.residual,
                cfg.ponder_max_steps, cfg.ponder_min_steps,
                cfg.ponder_prior_lambda, cfg.ponder_g_bias)
            self.ponder.apply(self._init_weights)

        self.last_signal: Optional[ControllerSignal] = None
        self.last_patch: Optional[PatchOutput] = None
        self.last_ponder: Optional[PonderInfo] = None
        self.last_moe_aux: Optional[torch.Tensor] = None    
        self.last_mtp: Optional[torch.Tensor] = None        

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor,
                gt_difficulty: Optional[torch.Tensor] = None,
                patch_aux_weight: float = 0.0,
                skip_ponder: bool = False) -> torch.Tensor:
        B, T = x.shape
        h = self.tok_emb(x) + self.pos(T, x.device)             
        h = self.drop(h)

        signal: Optional[ControllerSignal] = None
        if self.controller is not None:
            signal = self.controller(h)                          
            
            
            
            self.last_signal = signal

        if self.patcher is not None:                             
            g = signal.g if signal is not None else None
            pout = self.patcher(h, g=g, gt_difficulty=gt_difficulty,
                                aux_weight=patch_aux_weight)
            h = pout.h
            self.last_patch = pout

        for blk in self.blocks:
            h = blk(h, signal)

        if self.blocks[0].is_moe:                                
            self.last_moe_aux = torch.stack(
                [blk.mlp.last_info.aux_loss for blk in self.blocks]).sum()

        if self.memory is not None:                              
            h = self.memory(h, signal)

        if self.ponder is not None and not skip_ponder:          
            h, pinfo = self.ponder(h, signal)                    
            self.last_ponder = pinfo

        h = self.ln_f(h)
        if self.mtp_head is not None:                            
            self.last_mtp = self.mtp_head(h)
        return self.lm_head(h)                                   

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if hasattr(self.pos, "emb"):
                n -= self.pos.emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.tok_emb.weight.numel()
        return n


def build_model(model_cfg: Dict[str, Any],
                enable_controller: Optional[bool] = None,
                pos: Optional[str] = None,
                dyn_patch: Optional[bool] = None,
                attn_budget: Optional[bool] = None,
                residual: Optional[str] = None,
                latent_ponder: Optional[bool] = None,
                moe: Optional[bool] = None,
                memory: Optional[bool] = None,
                mtp: Optional[bool] = None) -> DenseDecoder:
    """Factory: dict (from YAML) -> DenseDecoder, with optional overrides."""
    cfg = ModelConfig.from_dict(model_cfg)
    if enable_controller is not None:
        cfg.enable_controller = enable_controller
    if pos is not None:
        cfg.pos = pos
    if dyn_patch is not None:
        cfg.dyn_patch = dyn_patch
    if attn_budget is not None:
        cfg.attn_budget = attn_budget
    if residual is not None:
        cfg.residual = residual
    if latent_ponder is not None:
        cfg.latent_ponder = latent_ponder
    if moe is not None:
        cfg.moe = moe
    if memory is not None:
        cfg.memory = memory
    if mtp is not None:
        cfg.mtp = mtp
    return DenseDecoder(cfg)


if __name__ == "__main__":
    
    torch.manual_seed(0)
    base = dict(vocab_size=260, d_model=384, n_layers=6, n_heads=6,
                d_ff=1536, max_seq_len=512)
    x = torch.randint(0, 260, (2, 64))

    torch.manual_seed(0)
    dense = build_model(base)
    torch.manual_seed(0)
    full = build_model(base, enable_controller=True, pos="nope", dyn_patch=True)
    ld = dense(x)
    lf = full(x, gt_difficulty=torch.zeros(2, 64), patch_aux_weight=1.0)
    assert ld.shape == lf.shape == (2, 64, 260)
    
    long_logits = full(torch.randint(0, 260, (2, 200)))
    print(f"model smoke OK | dense={dense.num_params()/1e6:.2f}M "
          f"full={full.num_params()/1e6:.2f}M | nope long-seq {tuple(long_logits.shape)} | "
          f"patch avg_len={64/full.last_patch.num_patches.float().mean():.2f}")
