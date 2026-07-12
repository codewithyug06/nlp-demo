from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PatchOutput:
    h: torch.Tensor
    boundary_prob: torch.Tensor
    num_patches: torch.Tensor
    aux_loss: Optional[torch.Tensor]

class DynamicPatcher(nn.Module):

    def __init__(self, d_model: int, *, hidden: int=64, ctx_weight: float=0.5, use_signal: bool=True):
        super().__init__()
        self.use_signal = use_signal
        in_dim = d_model + (1 if use_signal else 0)
        self.boundary_head = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.ctx_proj = nn.Linear(d_model, d_model)
        self.ctx_weight = ctx_weight

    def forward(self, h: torch.Tensor, g: Optional[torch.Tensor]=None, gt_difficulty: Optional[torch.Tensor]=None, aux_weight: float=0.0) -> PatchOutput:
        B, T, d = h.shape
        feat = h
        if self.use_signal:
            g_in = g if g is not None else torch.zeros(B, T, device=h.device)
            feat = torch.cat([h, g_in.unsqueeze(-1)], dim=-1)
        logit = self.boundary_head(feat).squeeze(-1)
        p = torch.sigmoid(logit)
        hard = p > 0.5
        hard[:, 0] = True
        idx = torch.arange(T, device=h.device).expand(B, T)
        masked_idx = torch.where(hard, idx, torch.full_like(idx, -1))
        start = torch.cummax(masked_idx, dim=1).values
        count = (idx - start + 1).clamp(min=1).unsqueeze(-1)
        cs = torch.cumsum(h, dim=1)
        cs_shift = torch.cat([torch.zeros(B, 1, d, device=h.device), cs[:, :-1]], dim=1)
        seg_sum = cs - torch.gather(cs_shift, 1, start.unsqueeze(-1).expand(B, T, d))
        ctx = seg_sum / count
        h_out = h + self.ctx_weight * self.ctx_proj(ctx)
        num_patches = hard.sum(dim=1)
        aux = None
        if gt_difficulty is not None and aux_weight > 0.0:
            target = gt_difficulty.float().clamp(0.0, 1.0).nan_to_num(0.0)
            aux = aux_weight * F.binary_cross_entropy_with_logits(logit, target)
        return PatchOutput(h_out, p, num_patches, aux)

def patch_stats(out: PatchOutput, difficulty: torch.Tensor, seq_len: int) -> dict:
    hard_mask = difficulty > 0
    p = out.boundary_prob
    avg_len = (seq_len / out.num_patches.float()).mean().item()
    return {'avg_patch_len': avg_len, 'p_boundary_hard': p[hard_mask].mean().item() if hard_mask.any() else float('nan'), 'p_boundary_easy': p[~hard_mask].mean().item(), 'mean_num_patches': out.num_patches.float().mean().item()}
if __name__ == '__main__':
    torch.manual_seed(0)
    B, T, d = (4, 32, 48)
    patcher = DynamicPatcher(d, use_signal=True)
    h = torch.randn(B, T, d)
    g = torch.rand(B, T)
    diff = torch.zeros(B, T)
    diff[:, T - 2] = 1.0
    out = patcher(h, g=g, gt_difficulty=diff, aux_weight=1.0)
    assert out.h.shape == (B, T, d)
    assert out.boundary_prob.shape == (B, T)
    h2 = h.clone()
    h2[:, T - 1] += 10.0
    out2 = patcher(h2, g=g)
    same_prefix = torch.allclose(out.h[:, :T - 2], out2.h[:, :T - 2], atol=1e-05)
    st = patch_stats(out, diff, T)
    print(f"dyn_patch smoke OK | causal_prefix_unchanged={same_prefix} | avg_patch_len={st['avg_patch_len']:.1f} p_hard={st['p_boundary_hard']:.2f} p_easy={st['p_boundary_easy']:.2f} aux={float(out.aux_loss):.3f}")
    assert same_prefix, 'context injection is NOT causal!'