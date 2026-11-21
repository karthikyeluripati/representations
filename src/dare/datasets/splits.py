"""Leakage-safe data splits.

ALFWorld already separates task instances into train / valid_seen (seen rooms)
/ valid_unseen (unseen rooms). We use:
  * fitting data (representations, predictors): drawn ONLY from the `train` split;
  * an internal train/dev split *by task instance* (all trajectories and decision
    states of one task_id fall on one side), optionally holding out whole task
    families for a family-transfer test;
  * evaluation: valid_seen (in-distribution rooms) and valid_unseen (OOD rooms).
"""
from __future__ import annotations

from dare.utils.common import stable_hash


def split_by_task(task_ids: list[str], dev_fraction: float, salt: str = "dare-v1",
                  holdout_families: tuple[str, ...] = ()) -> dict[str, str]:
    """Deterministically assign each task_id to 'fit', 'dev' or 'family_holdout'."""
    out = {}
    for tid in task_ids:
        fam = tid.split("-")[0]
        if fam in holdout_families:
            out[tid] = "family_holdout"
            continue
        u = int(stable_hash(salt, tid, n_hex=8), 16) / 0xFFFFFFFF
        out[tid] = "dev" if u < dev_fraction else "fit"
    return out


def assert_no_leakage(records_a: list[dict], records_b: list[dict], key: str = "task_id") -> None:
    overlap = {r[key] for r in records_a} & {r[key] for r in records_b}
    if overlap:
        raise AssertionError(f"{len(overlap)} {key}s appear in both partitions, e.g. {sorted(overlap)[:3]}")
