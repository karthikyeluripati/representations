"""Task construction and family-grouped splits (version cacrt_tasks_v1).

Split rule (fixed before any training):
  * every BFCL base task that involves VehicleControlAPI -> `heldout` (unseen tool family); all of its turns go there;
  * other base tasks keep the train/dev/test assignment of the previous project's v2 split (by base task id), so
    task pairs/turns never cross splits; base tasks absent from v2 are put in `unassigned` and not used.
Leakage checks: exact user-text hash and exact ground-truth-sequence hash must not be shared across splits.
"""
from __future__ import annotations

import collections
import hashlib
import json

from src.environment.bfcl_env import build_task, load_bfcl

HELDOUT_FAMILY = "VehicleControlAPI"
VERSION = "cacrt_tasks_v1"


def build_all_tasks(v2_split_by_base: dict):
    q, a, _ = load_bfcl()
    tasks, dropped = [], collections.Counter()
    for e in q:
        gt = a[e["id"]]
        for turn in range(len(gt)):
            t = build_task(e, gt, turn)
            if t is None:
                dropped["no_state_changing_call_or_unresolvable"] += 1
                continue
            if HELDOUT_FAMILY in e["involved_classes"]:
                t["split"] = "heldout"
            else:
                t["split"] = v2_split_by_base.get(e["id"], "unassigned")
                if t["split"] == "test_heldout_family":     # v2 put it in held-out without Vehicle: impossible by rule
                    t["split"] = "unassigned"
            tasks.append(t)
    return tasks, dropped


def _h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


def leakage_report(tasks):
    by_user, by_gt = collections.defaultdict(set), collections.defaultdict(set)
    for t in tasks:
        if t["split"] == "unassigned":
            continue
        by_user[_h(t["user"])].add(t["split"])
        by_gt[_h(t["turn_gt"])].add(t["split"])
    base_splits = collections.defaultdict(set)
    for t in tasks:
        base_splits[t["base_id"]].add(t["split"])
    return {"user_text_shared_across_splits": sum(len(s) > 1 for s in by_user.values()),
            "gt_sequence_shared_across_splits": sum(len(s) > 1 for s in by_gt.values()),
            "base_tasks_in_multiple_splits": sum(len(s) > 1 for s in base_splits.values())}
