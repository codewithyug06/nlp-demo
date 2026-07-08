"""STAGE 0 acceptance test (CLAUDE.md §5).

Checks:
  1. Every module in the scaffold imports cleanly.
  2. Two seeded random tensors are bit-identical (determinism harness works).
  3. All three configs load and the coupling flag is config-only (§2 rule 5).
  4. utils.write_run_json produces a valid provenance file.

Prints a single PASS/FAIL banner and exits non-zero on FAIL. Then STOP.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from utils import banner, get_logger, load_config, set_seed, write_run_json

log = get_logger("stage0")

PKG_MODULES = [
    "controller", "dyn_patch", "pos", "attn_budget", "residual", "halting",
    "latent_loop", "moe_router", "mem", "consolidate", "decode",
    "block", "model", "losses",
]
DATA_MODULES = ["synthetic", "scan", "lra"]
EVAL_MODULES = ["flops", "calibration", "faithfulness", "compositional",
                "length_gen", "forgetting"]
TOP_MODULES = ["utils", "train", "ablate"]

CONFIGS = ["configs/dense_baseline.yaml", "configs/decoupled.yaml",
           "configs/cortex_full.yaml"]


def check_imports() -> tuple[bool, str]:
    failed = []
    targets = (
        [f"cortex.{m}" for m in PKG_MODULES]
        + [f"data.{m}" for m in DATA_MODULES]
        + [f"eval.{m}" for m in EVAL_MODULES]
        + TOP_MODULES
    )
    for name in targets:
        try:
            importlib.import_module(name)
        except Exception as exc:  
            failed.append(f"{name}: {exc!r}")
    if failed:
        return False, "import failures: " + "; ".join(failed)
    return True, f"{len(targets)} modules imported"


def check_determinism() -> tuple[bool, str]:
    set_seed(1234)
    a_t = torch.randn(64, 64)
    a_n = np.random.randn(64)
    set_seed(1234)
    b_t = torch.randn(64, 64)
    b_n = np.random.randn(64)
    ok = torch.equal(a_t, b_t) and np.array_equal(a_n, b_n)
    return ok, ("torch+numpy draws bit-identical" if ok
                else "seeded draws differ — determinism broken")


def check_configs() -> tuple[bool, str]:
    modes = {}
    for path in CONFIGS:
        if not Path(path).exists():
            return False, f"missing config {path}"
        cfg = load_config(path)
        modes[path] = cfg.get("control_mode")
    want = {
        "configs/dense_baseline.yaml": "off",
        "configs/decoupled.yaml": "decoupled",
        "configs/cortex_full.yaml": "shared",
    }
    if modes != want:
        return False, f"control_mode mismatch: got {modes}"
    return True, f"coupling is config-only: {want}"


def check_run_json() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        p = write_run_json(
            Path(td) / "run.json",
            config={"probe": True},
            seed=0,
            metrics={"noop": 1.0},
        )
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        need = {"git_hash", "seed", "config", "metrics", "torch"}
        if not need.issubset(data):
            return False, f"run.json missing keys: {need - set(data)}"
    return True, "run.json schema ok"


def main() -> int:
    checks = [
        ("imports", check_imports),
        ("determinism", check_determinism),
        ("configs", check_configs),
        ("run.json", check_run_json),
    ]
    all_ok = True
    details = []
    for label, fn in checks:
        ok, msg = fn()
        all_ok &= ok
        log.info("[%s] %s :: %s", "ok " if ok else "ERR", label, msg)
        details.append(f"{label}={'ok' if ok else 'FAIL'}")

    banner(all_ok, "STAGE 0 — skeleton & determinism", " ".join(details))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
