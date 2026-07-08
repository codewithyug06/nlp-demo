"""CORTEX shared utilities.

Determinism harness + bookkeeping used by every stage:
  - set_seed(seed): seed python/numpy/torch/CUDA and force deterministic kernels.
  - load_config(path): the ONE YAML loader (no Hydra / no framework).
  - get_git_hash(): short git hash for provenance in run.json.
  - write_run_json(...): config snapshot + git hash + seed + final metrics.
  - get_logger(name): stdlib logger (stage PASS/FAIL banners are printed, not logged).

These are the primitives the acceptance tests lean on, so they must be boring and correct.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml





def set_seed(seed: int) -> None:
    """Seed every RNG CORTEX touches and force deterministic behaviour.

    Rule 7 (§2): every number must be reproducible by one command. We seed
    python `random`, numpy, and torch (CPU + all CUDA devices), and enable
    deterministic cuDNN. `use_deterministic_algorithms` is best-effort: some
    ops have no deterministic kernel, so we warn-only rather than hard-fail
    to keep the harness usable on CPU-only boxes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  
        pass





def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config into a plain dict. No merging magic, no Hydra."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping, got {type(cfg)}")
    return cfg





def get_git_hash() -> str:
    """Return the short git hash, or 'nogit' if not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def write_run_json(
    path: str | os.PathLike,
    config: Dict[str, Any],
    seed: int,
    metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the run.json provenance file required for every run (§3)."""
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": get_git_hash(),
        "seed": seed,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": config,
        "metrics": metrics,
    }
    if extra:
        payload["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
    return path





def get_logger(name: str = "cortex", level: int = logging.INFO) -> logging.Logger:
    """Stdlib logger. PASS/FAIL banners use print(), everything else logs here."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the training device; CUDA if available and preferred."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def banner(passed: bool, stage: str, detail: str = "") -> None:
    """Print the stage acceptance banner (the one thing that is NOT logged)."""
    tag = "PASS" if passed else "FAIL"
    line = f"[{tag}] {stage}"
    if detail:
        line += f" :: {detail}"
    print("=" * len(line))
    print(line)
    print("=" * len(line))


if __name__ == "__main__":
    
    set_seed(0)
    a = torch.randn(3)
    set_seed(0)
    b = torch.randn(3)
    assert torch.equal(a, b), "seeding not deterministic"
    log = get_logger("utils.smoke")
    log.info("git_hash=%s device=%s", get_git_hash(), get_device())
    banner(True, "utils smoke", f"identical draw {a.tolist()}")
