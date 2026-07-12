from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

def _scored_mask(batch) -> torch.Tensor:
    return batch.difficulty > 0

def lm_loss(logits: torch.Tensor, batch) -> torch.Tensor:
    mask = _scored_mask(batch)
    return F.cross_entropy(logits[mask], batch.y[mask])

def per_token_nll(logits: torch.Tensor, batch) -> torch.Tensor:
    mask = _scored_mask(batch)
    return F.cross_entropy(logits[mask], batch.y[mask], reduction='none').detach()

def get_batch_logps(logits: torch.Tensor, y: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
    mask = difficulty > 0
    nll = F.cross_entropy(logits.transpose(1, 2), y, reduction='none')
    logp = -nll
    return (logp * mask).sum(dim=1)

def mtp_loss(mtp_logits_list: list[torch.Tensor], batch) -> torch.Tensor:
    total_loss = 0.0
    valid_heads = 0
    for i, logits in enumerate(mtp_logits_list):
        pred = logits[:, :-(i + 1)]
        gold = batch.y[:, i + 1:]
        mask = _scored_mask(batch)[:, i + 1:]
        if mask.any():
            pred_masked = pred[mask]
            gold_masked = gold[mask]
            if torch.isnan(pred_masked).any():
                nan_loc = torch.where(torch.isnan(pred_masked))
                print(f'NAN LOC: {nan_loc}')
                import sys
                sys.exit(1)
            total_loss = total_loss + F.cross_entropy(pred_masked, gold_masked)
            valid_heads += 1
    return total_loss / max(valid_heads, 1)

def calibration_loss(logits: torch.Tensor, g_logit: torch.Tensor, batch) -> torch.Tensor:
    mask = _scored_mask(batch)
    nll = F.cross_entropy(logits[mask], batch.y[mask], reduction='none').detach()
    target = (1.0 - torch.exp(-nll)).clamp(0.0, 1.0).nan_to_num(0.0)
    g_logit_scored = g_logit[mask]
    return F.binary_cross_entropy_with_logits(g_logit_scored, target)

def ponder_cost(expected_steps: torch.Tensor) -> torch.Tensor:
    return expected_steps.mean()

def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor, beta: float=0.1) -> torch.Tensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()

def controller_dpo_penalty(g_chosen: torch.Tensor, g_rejected: torch.Tensor) -> torch.Tensor:
    return g_rejected.mean() - g_chosen.mean()
if __name__ == '__main__':
    from dataclasses import dataclass

    @dataclass
    class B:
        x: torch.Tensor
        y: torch.Tensor
        difficulty: torch.Tensor
        answer_pos: torch.Tensor
    torch.manual_seed(0)
    Bs, T, V = (4, 16, 32)
    x = torch.randint(0, V, (Bs, T))
    y = torch.randint(0, V, (Bs, T))
    diff = torch.zeros(Bs, T)
    diff[:, T - 2] = 1.0
    ap = torch.full((Bs,), T - 2)
    batch = B(x, y, diff, ap)
    logits = torch.randn(Bs, T, V, requires_grad=True)
    g = torch.rand(Bs, T)
    lm = lm_loss(logits, batch)
    mtp = mtp_loss([torch.randn(Bs, T, V) for _ in range(4)], batch)
    cal = calibration_loss(logits, g, batch)
    print(f'losses smoke OK | lm={lm.item():.3f} mtp={mtp.item():.3f} cal={cal.item():.3f} nll={per_token_nll(logits, batch).mean():.3f}')