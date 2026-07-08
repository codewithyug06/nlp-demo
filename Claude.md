# CORTEX — Master Build Prompt for Claude Code

> **The all-13 build.** One project, one spine, every transformer weakness addressed — nothing deferred to
> future work. Paste this whole file as the first message in a fresh Claude Code session in an empty repo.
>
> "In one shot" refers to **scope** (all 13 subsystems, no future-work bucket). It does **not** mean the agent
> writes everything in one turn — 13 coupled subsystems cannot be one-shotted correctly. The build is
> stage-gated so each subsystem is verified before the next. That is how you actually get all 13 working
> rather than 13 things that look plausible and silently break.

---

## 0. ROLE & MISSION

You are a senior research engineer building **CORTEX**: a full-stack adaptive decoder-only transformer whose
single organizing idea is a **Universal Compute Controller** — a tiny per-token module emitting one difficulty
signal `(g_t, c_t)` that governs *every* subsystem from input tokenization to output decoding.

The thesis: nearly all classic transformer weaknesses are symptoms of **uniform, unconditional compute**. One
calibrated per-token signal, routed everywhere, fixes the allocation policy once and pays off across the whole
stack.

You will implement 13 subsystems, each a designated consumer of the shared signal. Produce a **minimal,
reproducible, honestly-measured** PyTorch codebase — not a framework.

---

## 1. THE 13 SUBSYSTEMS (each has a slot; none is skipped)

| # | Weakness | Module | Consumes `(g_t,c_t)` to… |
|---|----------|--------|---------------------------|
| 1 | Attention span | `attn_budget.py` | set per-token key budget (differentiable top-k) |
| 2 | Positional / length-gen | `pos.py` | NoPE + adaptive depth for length generalization |
| 3 | Tokenization | `dyn_patch.py` | merge byte-patches; easy spans → fewer, larger patches |
| 4 | FFN / MoE | `moe_router.py` | scale active expert count with difficulty |
| 5 | Depth allocation | `halting.py` | PonderNet-style per-token halting |
| 6 | Memory / state | `mem.py` | gate reads/writes to a KV memory by uncertainty |
| 7 | Residual / norm | `residual.py` | structured residual + norm-placement study |
| 8 | Training objective | `losses.py` | LM + multi-token + ponder + calibration terms |
| 9 | Reasoning | `latent_loop.py` | extra latent refinement passes on hard tokens |
| 10 | Uncertainty | `controller.py` | the signal **is** a calibrated "I-don't-know" readout |
| 11 | Continual learning | `consolidate.py` | signal-gated consolidation to resist forgetting |
| 12 | Compositionality | `eval/compositional.py` | measured on SCAN/COGS/ListOps |
| 13 | Decoding | `decode.py` | difficulty-driven speculative/parallel decoding |

**The controller (`controller.py`) is the ONE new idea; the other 12 files are that idea applied to a
different part of the stack.**

---

## 2. NON-NEGOTIABLE RULES (STAGE DISCIPLINE)

1. Build in the **exact stage order in §5**. Never write a later stage while an earlier stage's acceptance
   test is unproven.
2. **End every stage: run its acceptance test, print PASS/FAIL, then STOP and report.** Wait for me to say
   "continue." Never chain stages silently.
3. **The dense baseline (Stage 1) must be trustworthy before any adaptive module exists.** Everything
   downstream is meaningless otherwise.
4. **The controller goes in inert first (Stage 2)** — compute the signal and log it, but let it change nothing
   — to prove the plumbing is neutral before it gets power.
5. **Every ablation is config-only.** The difference between "full CORTEX" and any ablation is a YAML file,
   never a forked code path. In particular, the **decoupled** control (13 private gates vs. 1 shared signal)
   must be a config flag.
6. Minimalist code: no framework beyond a YAML loader; no premature optimization; each file readable in one
   screen; a `__main__` smoke test per module.
7. Full determinism: seed `torch`/`numpy`/`random`/CUDA; every number reproducible by one command; log seed +
   git hash in `run.json`.
8. **Honesty over hype.** A subsystem that doesn't help still PASSES its stage if it runs correctly and the
   number is logged truthfully. If an acceptance test fails, diagnose it — never weaken the test to pass it.
9. Hardware target: **single RTX 5070 Ti (16 GB, CUDA 12.8)**. Every default config trains in hours. Model
   sizes 10M / 25M / 40M. Small is the point.
10. **Scope is fixed at all 13 — do not add a 14th, and do not silently drop one.** If a subsystem proves
    intractable at this scale, STOP and report it with a proposed minimal version; do not delete it.

---

## 3. TECH CONSTRAINTS

- Python 3.11+, PyTorch (CUDA 12.8), `pyyaml`, `numpy`, `tqdm`, `einops` (optional). Ask before adding more.
- No HF Trainer / Lightning / Hydra. One hand-written loop, one YAML loader.
- `bf16` mixed precision default; gradient checkpointing behind a flag (needed for the 40M + memory config).
- Every run writes `run.json` (config snapshot + git hash + seed + final metrics).

---

## 4. REPO LAYOUT (scaffold in Stage 0, fill over later stages)

```
cortex/
├── configs/
│   ├── dense_baseline.yaml
│   ├── decoupled.yaml            # 13 private gates (control for the coupling claim)
│   └── cortex_full.yaml        # 1 shared signal drives all 13
├── cortex/
│   ├── controller.py    # #10  shared signal (g_t, c_t)         ← CORE
│   ├── dyn_patch.py     # #3   dynamic byte-patch tokenization
│   ├── pos.py           # #2   NoPE / positional handling
│   ├── attn_budget.py   # #1   budgeted sparse attention
│   ├── residual.py      # #7   structured residual + norm
│   ├── halting.py       # #5   adaptive depth
│   ├── latent_loop.py   # #9   latent refinement reasoning
│   ├── moe_router.py    # #4   difficulty-scaled experts
│   ├── mem.py           # #6   gated KV memory (read + write)
│   ├── consolidate.py   # #11  signal-gated consolidation
│   ├── decode.py        # #13  difficulty-driven speculative decoding
│   ├── block.py         # wires all subsystems to ONE controller
│   ├── model.py         # assembles the decoder
│   └── losses.py        # #8   LM + multi-token + ponder + calibration
├── data/
│   ├── synthetic.py     # coref + needle (per-token difficulty KNOWN)
│   ├── scan.py          # #12 compositional
│   └── lra.py           # ListOps + retrieval
├── eval/
│   ├── flops.py
│   ├── calibration.py       # ECE, corr(g_t, NLL)
│   ├── faithfulness.py      # does g_t fire on ground-truth hard tokens?
│   ├── compositional.py     # #12
│   ├── length_gen.py        # #2 train len L, test 2L–4L
│   └── forgetting.py        # #11 sequential-task retention
├── train.py
├── ablate.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## 5. STAGES (each ends with STOP + report)

**STAGE 0 — Skeleton & determinism.** Repo layout (importable stubs with docstrings citing the subsystem #),
`utils.py` (seeding, run.json, logger), YAML loader, `requirements.txt`.
*Acceptance:* all modules import; two seeded random tensors are bit-identical. Print PASS/FAIL. **STOP.**

**STAGE 1 — Dense baseline + trustworthy loop.  ⚠️ HARD GATE.** Plain decoder in `model.py`, `train.py`,
`data/synthetic.py` (needle task with per-token difficulty labels), `eval/flops.py`.
*Acceptance:* 10M dense baseline beats chance on needle; `run.json` written; re-run reproduces within seed
variance. **STOP — nothing adaptive until this is trustworthy.**

**STAGE 2 — Universal Controller (#10), inert.** `controller.py` emits `(g_t, c_t)`; wire into `block.py` as a
logged no-op.
*Acceptance:* training matches Stage 1 within noise; `g_t` histograms logged. **STOP.**

**STAGE 3 — Input stack: #3 dynamic patching + #2 positional.** `dyn_patch.py` (byte-patch merge scaled by
`g_t`; keep a fixed-tokenizer fallback flag so it can't tank Stage-1 comparability) and `pos.py` (NoPE).
*Acceptance:* model trains with dynamic patching on; log avg patch length vs. difficulty; length-gen harness
runs. **STOP.**

**STAGE 4 — Attention stack: #1 budget + #7 residual/norm.** `attn_budget.py` (differentiable top-k,
temperature-annealed, straight-through fallback) + `residual.py` (structured residual, norm-placement flag).
*Acceptance (make-or-break):* on synthetic **coreference**, `eval/faithfulness.py` shows `g_t` spikes on
ground-truth anaphors above threshold. **STOP.**

**STAGE 5 — Compute stack: #5 halting + #9 latent loop.** `halting.py` (PonderNet + KL prior + min-passes
floor) and `latent_loop.py` (extra refinement passes on high-`g_t` tokens).
*Acceptance:* stable training; log avg ponder steps vs. difficulty; confirm no halting collapse. **STOP.**

**STAGE 6 — Knowledge stack: #4 MoE + #6 memory.** `moe_router.py` (difficulty-scaled expert count, ≤8
experts, load-balancing loss) + `mem.py` (uncertainty-gated read/write into residual stream).
*Acceptance:* balanced expert load (log imbalance); memory fires only on high-`g_t` tokens. **STOP.**

**STAGE 7 — Objective: #8.** `losses.py` = LM + multi-token-prediction head + ponder cost (`λ_c`) +
calibration term aligning `g_t` with detached per-token NLL.
*Acceptance:* full model trains to completion; `λ_c` sweep produces a quality-vs-FLOP frontier. **STOP.**

**STAGE 8 — Continual learning: #11.** `consolidate.py` — signal-gated consolidation (protect params/memory
slots the controller marks high-difficulty) across a 2–3 task sequence.
*Acceptance:* `eval/forgetting.py` shows retention on task A after training task B beats a no-consolidation
baseline (report honestly even if the gain is small). **STOP.**

**STAGE 9 — Decoding: #13.** `decode.py` — use per-token difficulty as a speculative-decoding acceptance
signal (easy tokens accepted in parallel, hard tokens verified).
*Acceptance:* generation matches greedy quality within tolerance while reducing sequential steps; log
speed/quality. **STOP.**

**STAGE 10 — Coupling test + full ablation grid (#12 included).** Build the **decoupled** config (13 private
gates). `ablate.py` runs: dense, decoupled, cortex_full, and per-subsystem leave-one-out, across 10M/25M.
Add `eval/compositional.py` (SCAN/COGS) and `eval/length_gen.py`.
*Acceptance:* `results/table.md` populated with real numbers + seeds + variance; frontier plot; the coupled
vs. decoupled comparison reported. **STOP.**

**STAGE 11 — README + reproducibility audit.** Every table number maps to one runnable command; clean-clone
dry run reproduces headline numbers.
*Acceptance:* audit passes. **STOP — final report.**

---

## 6. CODING CONVENTIONS

- Type hints + short docstrings citing the subsystem # and equation.
- Tensor shapes annotated at creation (`# (B, T, d)`).
- No magic numbers — everything tunable in YAML.
- Each module has a `__main__` smoke test (random in, shape-checked out).
- stdlib logger for everything except the stage PASS/FAIL banner.

## 7. WHEN UNSURE / WHEN A STAGE STRAINS

If a subsystem is intractable at this scale (likely candidates: #3 dynamic patching, #11 consolidation, #13
speculative decoding), **do NOT delete it.** STOP, state the blocker, and propose the smallest honest version
that still exercises the mechanism (e.g., a reduced task, fewer experts, a 2-task continual sequence). I decide
whether to accept the reduced version. Log every such choice in `run.json`.

## 8. FIRST ACTION

Begin **Stage 0 only.** Scaffold the repo and determinism harness, run the Stage 0 acceptance test, print
PASS/FAIL, and STOP for my confirmation.
