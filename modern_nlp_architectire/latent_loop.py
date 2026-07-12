from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .block import TransformerBlock
from .controller import ControllerSignal
from .halting import HaltingHead, PonderInfo, expected_steps, kl_geometric, ponder_probs

class LatentPonder(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float=0.0, norm: str='pre', *, attn_budget: bool=False, span_min: int=8, span_temp: float=4.0, residual: str='vanilla', max_steps: int=4, min_steps: int=1, prior_lambda: float=0.5, g_bias: float=2.0):
        super().__init__()
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.prior_lambda = prior_lambda
        self.g_bias = g_bias
        self.refiner = TransformerBlock(d_model, n_heads, d_ff, dropout=dropout, norm=norm, attn_budget=attn_budget, span_min=span_min, span_temp=span_temp, residual=residual)
        self.halt = HaltingHead(d_model)

    def forward(self, h: torch.Tensor, signal: Optional[ControllerSignal]=None) -> Tuple[torch.Tensor, PonderInfo]:
        g = signal.g if signal is not None else None
        state = h
        states, logits = ([], [])
        for _ in range(self.max_steps):
            state = self.refiner(state, signal)
            states.append(state)
            logits.append(self.halt(state, g, self.g_bias))
        halt_logits = torch.stack(logits, dim=0)
        p = ponder_probs(halt_logits, self.min_steps)
        S = torch.stack(states, dim=2)
        h_out = (p.unsqueeze(-1) * S).sum(dim=2)
        info = PonderInfo(p=p, expected_steps=expected_steps(p), kl=kl_geometric(p, self.prior_lambda))
        return (h_out, info)
if __name__ == '__main__':
    torch.manual_seed(0)
    B, T, d = (2, 16, 48)
    lp = LatentPonder(d, n_heads=4, d_ff=96, max_steps=4, min_steps=1, g_bias=3.0)
    h = torch.randn(B, T, d)
    g = torch.zeros(B, T)
    g[:, ::2] = 1.0
    sig = ControllerSignal(g=g, c=torch.zeros(B, T), logits=torch.zeros(B, T, 2))
    out, info = lp(h, sig)
    assert out.shape == (B, T, d)
    hard_steps = info.expected_steps[g > 0.5].mean().item()
    easy_steps = info.expected_steps[g <= 0.5].mean().item()
    print(f'latent_loop smoke OK | out {tuple(out.shape)} | E[steps] hard={hard_steps:.2f} easy={easy_steps:.2f} (hard should be >= easy) | KL={info.kl.mean():.3f}')
    assert hard_steps >= easy_steps, "g_bias failed: hard tokens didn't ponder more"