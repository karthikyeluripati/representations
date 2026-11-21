"""Seeding, hashing, config loading."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def stable_hash(*parts: Any, n_hex: int = 16) -> str:
    """Process-independent hash (Python's hash() is salted per process)."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:n_hex]


def derive_seed(*parts: Any) -> int:
    """Derive a 31-bit integer seed deterministically from arbitrary parts."""
    return int(stable_hash(*parts, n_hex=8), 16) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config(*paths: str | Path) -> dict:
    """Load and shallow-merge YAML configs (later files override earlier keys)."""
    cfg: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def config_hash(cfg: dict) -> str:
    return stable_hash(json.dumps(cfg, sort_keys=True, default=str))
