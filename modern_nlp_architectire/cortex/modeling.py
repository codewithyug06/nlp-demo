from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from .configuration import CortexConfig
from .block import TransformerBlock, RMSNorm
from .controller import Controller, ControllerSignal
from .dyn_patch import DynamicPatcher, PatchOutput
from .halting import PonderInfo
from .latent_loop import LatentPonder
from .mem import GatedKVMemory
from .pos import build_positional

class CortexPreTrainedModel(PreTrainedModel):
    config_class = CortexConfig
    base_model_prefix = 'cortex'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.0
    norm: str = 'pre'
    tie_embeddings: bool = True
    enable_controller: bool = False
    pos: str = 'learned'
    dyn_patch: bool = False
    patch_hidden: int = 64
    patch_ctx_weight: float = 0.5
    patch_use_signal: bool = True
    attn_budget: bool = False
    span_min: int = 8
    span_temp: float = 4.0
    residual: str = 'vanilla'
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
    def from_dict(cls, m: Dict[str, Any]) -> 'ModelConfig':
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in m.items() if k in fields})

class CortexForCausalLM(CortexPreTrainedModel):
    config_class = CortexConfig
    _tied_weights_keys = []

    def __init__(self, config: CortexConfig):
        super().__init__(config)
        pcfg = config.cortex_cfg
        model_cfg = pcfg.get('model', {})
        subsys_cfg = pcfg.get('subsystems', {})
        patch_cfg = pcfg.get('patch', {})
        obj_cfg = pcfg.get('objective', {})
        self.vocab_size = config.vocab_size
        self.cfg = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = build_positional(model_cfg.get('pos', 'nope'), config.max_seq_len, config.d_model, d_head=config.d_model // config.n_heads, rope_theta=config.rope_theta)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.n_heads, config.n_kv_heads, config.d_ff, dropout=config.dropout, norm=config.norm, attn_budget=subsys_cfg.get('attn_budget', False), span_min=model_cfg.get('span_min', 4), span_temp=model_cfg.get('span_temp', 1.0), residual=model_cfg.get('residual', 'vanilla'), moe=subsys_cfg.get('moe', False), moe_experts=model_cfg.get('moe_experts', 4), moe_k_max=model_cfg.get('moe_k_max', 2), moe_expert_ff=model_cfg.get('moe_expert_ff', 0), sliding_window=config.sliding_window, quantize_kv=config.quantize_kv) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        if config.mtp_depth > 0:
            self.mtp_norm = RMSNorm(config.d_model)
            from cortex.block import ResidualScaler, MLP
            self.mtp_head = nn.ModuleList([MLP(config.d_model, config.d_ff) for _ in range(config.mtp_depth)])
            self.mtp_res = nn.ModuleList([ResidualScaler(config.d_model, 'layerscale') for _ in range(config.mtp_depth)])
        else:
            self.mtp_head = None
            self.mtp_res = None
        self.vision_encoder = None
        if subsys_cfg.get('vision', False):
            from .vision import CortexVisionEncoder
            self.vision_encoder = CortexVisionEncoder(config.d_model, image_size=224, patch_size=16)
        self.controller: Optional[Controller] = None
        if subsys_cfg.get('controller', False):
            self.controller = Controller(config.d_model)
            self.controller = Controller(config.d_model)
        self.patcher: Optional[DynamicPatcher] = None
        if subsys_cfg.get('dyn_patch', False):
            self.patcher = DynamicPatcher(config.d_model, hidden=patch_cfg.get('patch_hidden', 32), ctx_weight=patch_cfg.get('patch_ctx_weight', 0.5))
        self.latent_ponder: Optional[LatentPonder] = None
        if subsys_cfg.get('latent_loop', False):
            self.latent_ponder = LatentPonder(config.d_model, config.n_heads, config.d_ff, dropout=config.dropout, norm=config.norm, attn_budget=subsys_cfg.get('attn_budget', False), span_min=model_cfg.get('span_min', 4), span_temp=model_cfg.get('span_temp', 1.0), residual=model_cfg.get('residual', 'vanilla'), max_steps=model_cfg.get('ponder_max_steps', 4), min_steps=model_cfg.get('ponder_min_steps', 1))
        self.mem: Optional[GatedKVMemory] = None
        if subsys_cfg.get('memory', False):
            self.mem = GatedKVMemory(config.d_model, n_slots=model_cfg.get('mem_slots', 32))
            self.mem = GatedKVMemory(config.d_model, n_slots=model_cfg.get('mem_slots', 32))
        self.last_signal: Optional[ControllerSignal] = None
        self.last_patch: Optional[PatchOutput] = None
        self.last_ponder: Optional[PonderInfo] = None
        self.last_moe_aux: Optional[torch.Tensor] = None
        self.last_mtp: Optional[list[torch.Tensor]] = None
        self.post_init()

    def forward(self, x: torch.Tensor, pixel_values: Optional[torch.Tensor]=None, gt_difficulty: Optional[torch.Tensor]=None, patch_aux_weight: float=1.0, skip_ponder: bool=False, use_cache: bool=False, past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]]=None):
        B, T = x.shape
        start_pos = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        h = self.tok_emb(x)
        if self.vision_encoder is not None and pixel_values is not None:
            v_emb = self.vision_encoder(pixel_values)
            h = torch.cat((v_emb, h), dim=1)
        T_total = h.shape[1]
        pos_emb = self.pos(T_total, x.device, start_pos=start_pos)
        freqs_cis = None
        if self.cfg.cortex_cfg.get('model', {}).get('pos', 'nope') == 'rope':
            freqs_cis = pos_emb
        else:
            h = h + pos_emb
        h = self.drop(h)
        signal: Optional[ControllerSignal] = None
        if self.controller is not None:
            signal = self.controller(h)
            self.last_signal = signal
        if self.patcher is not None:
            g = signal.g if signal is not None else None
            pout = self.patcher(h, g=g, gt_difficulty=gt_difficulty, aux_weight=patch_aux_weight)
            h = pout.h
            self.last_patch = pout
        next_cache = () if use_cache else None
        for i, blk in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self, 'gradient_checkpointing', False) and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, signal, use_cache, past_key_value, freqs_cis, use_reentrant=False)
            elif use_cache:
                h, present_key_value = blk(h, signal, use_cache=True, past_key_value=past_key_value, freqs_cis=freqs_cis)
                next_cache += (present_key_value,)
            else:
                h = blk(h, signal, freqs_cis=freqs_cis)
        if self.blocks[0].is_moe:
            self.last_moe_aux = torch.stack([blk.mlp.last_info.aux_loss for blk in self.blocks]).sum()
        if gt_difficulty is not None:
            g_mask = gt_difficulty > 0
            g_mask = g_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, T, -1).clone()
            idx = torch.arange(T, device=x.device)
            g_mask[:, :, idx, idx] = True
        if self.mem is not None:
            h = self.mem(h, signal)
        if self.latent_ponder is not None and (not skip_ponder):
            h, pinfo = self.latent_ponder(h, signal)
            self.last_ponder = pinfo
        h = self.ln_f(h)
        if self.mtp_head is not None:
            mtp_list = []
            mtp_feat = h
            for mlp, res in zip(self.mtp_head, self.mtp_res):
                mtp_feat = mtp_feat + res(mlp(self.mtp_norm(mtp_feat)))
                mtp_list.append(self.lm_head(mtp_feat))
            self.last_mtp = mtp_list
        logits = self.lm_head(h)
        if use_cache:
            return (logits, next_cache)
        return logits

    def tie_weights(self, **kwargs):
        if getattr(self.config, 'tie_embeddings', False):
            self.lm_head.weight = self.tok_emb.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb

    def num_params(self, non_embedding: bool=False) -> int:
        n = sum((p.numel() for p in self.parameters()))
        if non_embedding:
            if hasattr(self.pos, 'emb'):
                n -= self.pos.emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.tok_emb.weight.numel()
        return n

def build_model(model_cfg: Dict[str, Any], enable_controller: Optional[bool]=None, pos: Optional[str]=None, dyn_patch: Optional[bool]=None, attn_budget: Optional[bool]=None, residual: Optional[str]=None, latent_ponder: Optional[bool]=None, moe: Optional[bool]=None, memory: Optional[bool]=None, mtp: Optional[bool]=None) -> DenseDecoder:
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
if __name__ == '__main__':
    torch.manual_seed(0)
    base = dict(vocab_size=260, d_model=384, n_layers=6, n_heads=6, d_ff=1536, max_seq_len=512)
    x = torch.randint(0, 260, (2, 64))
    torch.manual_seed(0)
    dense = build_model(base)
    torch.manual_seed(0)
    full = build_model(base, enable_controller=True, pos='nope', dyn_patch=True)
    ld = dense(x)
    lf = full(x, gt_difficulty=torch.zeros(2, 64), patch_aux_weight=1.0)
    assert ld.shape == lf.shape == (2, 64, 260)
    long_logits = full(torch.randint(0, 260, (2, 200)))
    print(f'model smoke OK | dense={dense.num_params() / 1000000.0:.2f}M full={full.num_params() / 1000000.0:.2f}M | nope long-seq {tuple(long_logits.shape)} | patch avg_len={64 / full.last_patch.num_patches.float().mean():.2f}')