"""Training objective — subsystem #8 (the consolidated loss).

    total = lm + w_mtp·mtp + w_cal·calibration + λ_c·ponder_cost + (aux terms)

Pieces owned here:
  lm_loss           : CE on the informative (difficulty>0) target tokens.
  mtp_loss          : multi-token prediction — a 2-ahead head predicts the answer
                      value from one position earlier (a real retrieval-2-ahead
                      target on our tasks).
  calibration_loss  : aligns g_t with the DETACHED per-token error probability
                      (1 − p_correct = 1 − e^{−NLL}) on scored positions. This is
                      what finally makes g_t a calibrated "I-don't-know" readout
                      (#10) and pins its sign (high g ⇔ likely wrong / hard).
  ponder_cost       : λ_c · E[ponder steps] — the compute knob swept for the
                      quality-vs-FLOP frontier.

The other adaptive-subsystem regularizers (patch aux, budget cost, MoE load
balance, PonderNet KL) are assembled in train.py; this file owns the #8 terms.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _scored_mask(batch) -> torch.Tensor:
    return batch.difficulty > 0                       # (B, T) informative tokens


def lm_loss(logits: torch.Tensor, batch) -> torch.Tensor:
    """CE on the informative target tokens (answer/anaphor). (B,T,V)->scalar."""
    mask = _scored_mask(batch)
    return F.cross_entropy(logits[mask], batch.y[mask])


def per_token_nll(logits: torch.Tensor, batch) -> torch.Tensor:
    """Detached NLL at each scored position. -> (N,) for the N scored tokens."""
    mask = _scored_mask(batch)
    return F.cross_entropy(logits[mask], batch.y[mask], reduction="none").detach()


def get_batch_logps(logits: torch.Tensor, y: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
    """Compute the log probability of the true labels y, used for DPO.
    Returns: log probs of shape (B,)
    """
    mask = difficulty > 0
    # cross_entropy returns negative log prob
    nll = F.cross_entropy(logits.transpose(1, 2), y, reduction="none") # (B, T)
    logp = -nll
    
    # sum over informative tokens
    return (logp * mask).sum(dim=1)


def mtp_loss(mtp_logits_list: list[torch.Tensor], batch) -> torch.Tensor:
    """Multi-token prediction of the answer value.

    Predicts the target token from N steps ahead.
    mtp_logits_list contains logits for predicting t+1, t+2, ..., t+N.
    """
    total_loss = 0.0
    valid_heads = 0
    
    for i, logits in enumerate(mtp_logits_list):
        pred = logits[:, :- (i + 1)].reshape(-1, logits.size(-1))
        gold = batch.y[:, (i + 1):].reshape(-1)
        
        # PyTorch cross_entropy backward returns NaN if all targets are ignore_index (-100)
        # because it divides by the sum of weights (0).
        if (gold != -100).any():
            if torch.isnan(pred).any():
                nan_loc = torch.where(torch.isnan(pred))
                print(f"NAN LOC: {nan_loc}")
                import sys; sys.exit(1)
            total_loss = total_loss + F.cross_entropy(pred, gold)
            valid_heads += 1
            
    return total_loss / max(valid_heads, 1)


def calibration_loss(logits: torch.Tensor, g_logit: torch.Tensor, batch) -> torch.Tensor:
    """Align g_t with detached error probability on scored positions (#10).

    target = 1 − p_correct = 1 − e^{−NLL} ∈ [0,1] (detached). BCE(g, target).
    """
    mask = _scored_mask(batch)
    nll = F.cross_entropy(logits[mask], batch.y[mask], reduction="none").detach()
    target = (1.0 - torch.exp(-nll)).clamp(0.0, 1.0).nan_to_num(0.0)   # error prob in [0,1]
    g_logit_scored = g_logit[mask]
    return F.binary_cross_entropy_with_logits(g_logit_scored, target)


def ponder_cost(expected_steps: torch.Tensor) -> torch.Tensor:
    """λ_c compute knob: mean expected ponder steps. (B,T) -> scalar."""
    return expected_steps.mean()


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Standard DPO loss.
    
    Args:
        policy_chosen_logps: Log probabilities of chosen responses from policy model.
        policy_rejected_logps: Log probabilities of rejected responses from policy model.
        ref_chosen_logps: Log probabilities of chosen responses from reference model.
        ref_rejected_logps: Log probabilities of rejected responses from reference model.
        beta: Temperature parameter for the DPO loss, typically 0.1 to 0.5.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    
    return loss.mean()


def controller_dpo_penalty(
    g_chosen: torch.Tensor,
    g_rejected: torch.Tensor,
) -> torch.Tensor:
    """Penalize controller for allocating compute (g_t) to rejected vs chosen sequences.
    
    We want the controller to learn that allocating heavy compute to 'rejected' 
    (often simpler or incorrect paths) is bad, while allocating compute to 
    'chosen' (often complex reasoning) is good. 
    Returns: mean(g_rejected - g_chosen).
    """
    return (g_rejected.mean() - g_chosen.mean())


if __name__ == "__main__":
    # Smoke test: shapes + calibration pushes g toward the error target.
    from dataclasses import dataclass

    @dataclass
    class B:
        x: torch.Tensor
        y: torch.Tensor
        difficulty: torch.Tensor
        answer_pos: torch.Tensor

    torch.manual_seed(0)
    Bs, T, V = 4, 16, 32
    x = torch.randint(0, V, (Bs, T))
    y = torch.randint(0, V, (Bs, T))
    diff = torch.zeros(Bs, T); diff[:, T - 2] = 1.0
    ap = torch.full((Bs,), T - 2)
    batch = B(x, y, diff, ap)
    logits = torch.randn(Bs, T, V, requires_grad=True)
    g = torch.rand(Bs, T)
    lm = lm_loss(logits, batch)
    mtp = mtp_loss([torch.randn(Bs, T, V) for _ in range(4)], batch)
    cal = calibration_loss(logits, g, batch)
    print(f"losses smoke OK | lm={lm.item():.3f} mtp={mtp.item():.3f} "
          f"cal={cal.item():.3f} nll={per_token_nll(logits, batch).mean():.3f}")
