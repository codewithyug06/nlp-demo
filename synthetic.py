"""Difficulty-labelled synthetic tasks — the backbone of the honesty story.

The whole PANOPTES thesis rests on a per-token difficulty signal, so we need
tasks where the ground-truth difficulty of each token is KNOWN, not guessed.

NEEDLE (this file, Stage 1): a retrieval / copy task.
    BOS  filler...  NEEDLE  V  filler...  QUERY  V
The value V appears once right after a NEEDLE marker at a RANDOM position, and
must be reproduced right after the QUERY marker at the end. Every other target
is either structural (predictable) or random filler (unpredictable). Only the
answer token requires long-range retrieval, so its ground-truth difficulty = 1;
all others = 0. Chance retrieval accuracy = 1 / n_values.

Vocabulary layout (fits vocab_size >= 4 + n_values, e.g. 260):
    0            PAD
    1            BOS
    2            NEEDLE
    3            QUERY
    [4, 4+nv)    value tokens
    [4+nv, V)    filler tokens

`make_batch` is fully driven by a torch.Generator so runs are reproducible
(Rule 7, §2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

PAD, BOS, NEEDLE, QUERY = 0, 1, 2, 3
N_SPECIAL = 4


@dataclass
class NeedleSpec:
    vocab_size: int = 260
    n_values: int = 32          # number of distinct value tokens (chance = 1/nv)
    seq_len: int = 256
    direction: str = "after"    # value sits 'after' or 'before' the NEEDLE marker
                                # (two directions => conflicting tasks for #11)
    value_range: str = "distinct"  # 'distinct': value from its own range (uniquely
                                # identifiable) | 'filler': value looks like filler,
                                # so the model MUST use marker+direction (needed for
                                # #11 so tasks A/B genuinely conflict)

    @property
    def value_lo(self) -> int:
        return N_SPECIAL

    @property
    def value_hi(self) -> int:
        return N_SPECIAL + self.n_values

    @property
    def filler_lo(self) -> int:
        return self.value_hi

    @property
    def filler_hi(self) -> int:
        return self.vocab_size

    @property
    def sample_lo(self) -> int:
        return self.filler_lo if self.value_range == "filler" else self.value_lo

    @property
    def sample_hi(self) -> int:
        return self.filler_hi if self.value_range == "filler" else self.value_hi

    @property
    def chance(self) -> float:
        return 1.0 / (self.sample_hi - self.sample_lo)

    def __post_init__(self) -> None:
        if self.filler_hi - self.filler_lo < 2:
            raise ValueError(
                f"vocab_size={self.vocab_size} too small for n_values={self.n_values}"
            )
        if self.seq_len < 8:
            raise ValueError("seq_len must be >= 8 for the needle layout")


@dataclass
class NeedleBatch:
    x: torch.Tensor           # (B, T) input token ids
    y: torch.Tensor           # (B, T) next-token targets (x shifted left; last = PAD)
    difficulty: torch.Tensor  # (B, T) 1.0 on the answer target position, else 0.0
    answer_pos: torch.Tensor  # (B,) index t where y[b, t] is the retrieved value
    needle_pos: torch.Tensor  # (B,) index of the NEEDLE marker (for faithfulness later)


def make_batch(
    batch_size: int,
    spec: NeedleSpec,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str = "cpu",
) -> NeedleBatch:
    """Generate one reproducible batch of the needle task. Shapes: (B, T)."""
    B, T = batch_size, spec.seq_len
    g = generator

    def randint(lo: int, hi: int, size) -> torch.Tensor:
        return torch.randint(lo, hi, size, generator=g)

    # (B, T) start as random filler everywhere.
    x = randint(spec.filler_lo, spec.filler_hi, (B, T))
    x[:, 0] = BOS

    # value V per row, drawn from the configured range. (B,)
    values = randint(spec.sample_lo, spec.sample_hi, (B,))

    # QUERY sits at T-2, answer V at T-1 (last position).
    x[:, T - 2] = QUERY
    x[:, T - 1] = values

    # NEEDLE marker at a random position; value sits just after OR before it,
    # depending on spec.direction (the two directions conflict -> #11 forgetting).
    needle_pos = randint(3, T - 4, (B,))                 # (B,) room on both sides
    rows = torch.arange(B)
    x[rows, needle_pos] = NEEDLE
    off = 1 if spec.direction == "after" else -1
    x[rows, needle_pos + off] = values

    # Next-token targets: y[t] = x[t+1]; last target is PAD (unused / not scored).
    y = torch.full((B, T), PAD, dtype=torch.long)
    y[:, : T - 1] = x[:, 1:]

    # Difficulty: only the answer target (predicting x[T-1] from position T-2).
    difficulty = torch.zeros((B, T), dtype=torch.float32)
    answer_pos = torch.full((B,), T - 2, dtype=torch.long)  # (B,)
    difficulty[rows, answer_pos] = 1.0

    return NeedleBatch(
        x=x.to(device),
        y=y.to(device),
        difficulty=difficulty.to(device),
        answer_pos=answer_pos.to(device),
        needle_pos=needle_pos.to(device),
    )


# --------------------------------------------------------------------------- #
# COREFERENCE (Stage 4): keyed retrieval with distractors.
# --------------------------------------------------------------------------- #
# Several (KEY, key_id, value) pairs are scattered through the sequence. At the
# end a QUERY names one key_id; the model must resolve it to the MATCHING pair's
# value (an anaphor -> antecedent link) while ignoring distractor pairs. The
# anaphor (the query answer position) is the one hard token: difficulty = 1.
# Because the matching antecedent sits at a random, often-distant position, the
# anaphor must attend far back -> it must raise g_t (see attn_budget.py #1).
#
# Vocab layout: 0 PAD, 1 BOS, 2 KEY_MARK, 3 QUERY,
#   [4, 4+nk)            key-id tokens
#   [4+nk, 4+nk+nv)      value tokens
#   [4+nk+nv, V)         filler
KEY_MARK = 2  # note: shares slot 2/3 with NEEDLE/QUERY; tasks are used separately


@dataclass
class CorefSpec:
    vocab_size: int = 260
    n_keys: int = 16
    n_values: int = 32
    n_pairs: int = 4
    seq_len: int = 128

    @property
    def key_lo(self) -> int:
        return N_SPECIAL

    @property
    def value_lo(self) -> int:
        return N_SPECIAL + self.n_keys

    @property
    def filler_lo(self) -> int:
        return N_SPECIAL + self.n_keys + self.n_values

    @property
    def chance(self) -> float:
        return 1.0 / self.n_values

    def __post_init__(self) -> None:
        if self.n_keys < self.n_pairs:
            raise ValueError("n_keys must be >= n_pairs (keys are distinct per seq)")
        if self.vocab_size - self.filler_lo < 2:
            raise ValueError("vocab too small for coref ranges")
        if self.seq_len < 6 + 3 * self.n_pairs:
            raise ValueError("seq_len too small for n_pairs blocks + query tail")


@dataclass
class CorefBatch:
    x: torch.Tensor           # (B, T)
    y: torch.Tensor           # (B, T) next-token targets
    difficulty: torch.Tensor  # (B, T) 1.0 at the anaphor (answer) position
    answer_pos: torch.Tensor  # (B,) target index of the resolved value
    antecedent_pos: torch.Tensor  # (B,) position of the matching value (the referent)


def make_coref_batch(batch_size: int, spec: CorefSpec,
                     generator: Optional[torch.Generator] = None,
                     device: torch.device | str = "cpu") -> CorefBatch:
    """One reproducible batch of the coreference task. Shapes: (B, T)."""
    B, T, P = batch_size, spec.seq_len, spec.n_pairs
    g = generator

    def randint(lo, hi, size):
        return torch.randint(lo, hi, size, generator=g)

    x = randint(spec.filler_lo, spec.vocab_size, (B, T))
    x[:, 0] = BOS
    rows = torch.arange(B)

    # Distinct key-ids per row (permute the key range, take first P).
    keys = torch.empty(B, P, dtype=torch.long)
    for b in range(B):
        perm = torch.randperm(spec.n_keys, generator=g)[:P]
        keys[b] = spec.key_lo + perm
    values = randint(spec.value_lo, spec.value_lo + spec.n_values, (B, P))

    # Place P non-overlapping [KEY_MARK, key, value] blocks in disjoint slots.
    region = T - 6                                  # leave tail for the query
    slot = region // P
    val_pos = torch.empty(B, P, dtype=torch.long)
    for p in range(P):
        base = 1 + p * slot
        off = randint(0, max(1, slot - 3), (B,))
        start = base + off                          # (B,)
        x[rows, start] = KEY_MARK
        x[rows, start + 1] = keys[:, p]
        x[rows, start + 2] = values[:, p]
        val_pos[:, p] = start + 2

    # Query tail: QUERY, key_q, answer(value_q).
    q = randint(0, P, (B,))                          # which pair is queried
    key_q = keys[rows, q]
    ans_val = values[rows, q]
    x[:, T - 3] = QUERY
    x[:, T - 2] = key_q
    x[:, T - 1] = ans_val

    y = torch.full((B, T), PAD, dtype=torch.long)
    y[:, : T - 1] = x[:, 1:]
    difficulty = torch.zeros((B, T), dtype=torch.float32)
    answer_pos = torch.full((B,), T - 2, dtype=torch.long)
    difficulty[rows, answer_pos] = 1.0

    return CorefBatch(
        x=x.to(device), y=y.to(device), difficulty=difficulty.to(device),
        answer_pos=answer_pos.to(device),
        antecedent_pos=val_pos[rows, q].to(device),
    )


def retrieval_accuracy(logits: torch.Tensor, batch) -> float:
    """Fraction of rows whose answer token is predicted correctly.

    logits: (B, T, V). We read the prediction at each row's answer_pos and
    compare argmax to the true value target y[b, answer_pos].
    """
    B = logits.shape[0]
    rows = torch.arange(B, device=logits.device)
    ap = batch.answer_pos
    pred = logits[rows, ap].argmax(dim=-1)      # (B,)
    gold = batch.y[rows, ap]                     # (B,)
    return (pred == gold).float().mean().item()


if __name__ == "__main__":
    # Smoke test: shapes, label placement, determinism.
    spec = NeedleSpec(vocab_size=260, n_values=32, seq_len=64)
    g1 = torch.Generator().manual_seed(0)
    b1 = make_batch(8, spec, g1)
    g2 = torch.Generator().manual_seed(0)
    b2 = make_batch(8, spec, g2)
    assert torch.equal(b1.x, b2.x), "batch not reproducible"
    assert b1.x.shape == (8, 64)
    # Exactly one difficulty==1 per row, at answer_pos, and it marks the value.
    assert torch.equal(b1.difficulty.sum(dim=1), torch.ones(8))
    rows = torch.arange(8)
    assert torch.equal(b1.y[rows, b1.answer_pos], b1.x[rows, spec.seq_len - 1])
    print(f"needle smoke OK | chance={spec.chance:.4f} | "
          f"x[0,-4:]={b1.x[0, -4:].tolist()} answer_pos={b1.answer_pos[0].item()}")

    # Coref smoke: determinism, one hard token, answer == matching antecedent value.
    cspec = CorefSpec(vocab_size=260, n_keys=16, n_values=32, n_pairs=4, seq_len=64)
    cg1 = torch.Generator().manual_seed(1)
    c1 = make_coref_batch(8, cspec, cg1)
    cg2 = torch.Generator().manual_seed(1)
    c2 = make_coref_batch(8, cspec, cg2)
    assert torch.equal(c1.x, c2.x), "coref not reproducible"
    assert torch.equal(c1.difficulty.sum(dim=1), torch.ones(8))
    r = torch.arange(8)
    # The resolved answer equals the value stored at the antecedent position.
    assert torch.equal(c1.y[r, c1.answer_pos], c1.x[r, c1.antecedent_pos])
    print(f"coref smoke OK | chance={cspec.chance:.4f} n_pairs={cspec.n_pairs} | "
          f"antecedent_pos[0]={c1.antecedent_pos[0].item()} "
          f"answer_pos[0]={c1.answer_pos[0].item()}")
