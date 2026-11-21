"""Evaluation episode suites (fixed before training; test and heldout splits are never used for training or selection).

  S0                    fault-free, 1 seed (deterministic)
  S1, S2, S3            injected outages, v2 error pool, seeds 0-2
  S1_obs, S2_obs        observation shift: reserved EVAL error strings (never in training data), seed 0
  S2_act                action shift: alternative endpoint uses the reserved `<f>_replica` naming, seed 0
  RP, RA                genuine-error repair: forced prefix ending in a REAL BFCL error, from verified eval states
"""
from __future__ import annotations

import json


def build_specs(tasks, eval_states, splits=("test", "heldout")):
    specs = []
    for t in tasks:
        if not t["usable"] or t["split"] not in splits:
            continue
        base = {"task_id": t["id"], "pool": "v2", "alt": "train"}
        specs.append({**base, "suite": "S0", "scenario": "S0", "seed": 0})
        for sc in ("S1", "S2", "S3"):
            for sd in (0, 1, 2):
                specs.append({**base, "suite": sc, "scenario": sc, "seed": sd})
        specs.append({**base, "suite": "S1_obs", "scenario": "S1", "seed": 0, "pool": "eval"})
        specs.append({**base, "suite": "S2_obs", "scenario": "S2", "seed": 0, "pool": "eval"})
        specs.append({**base, "suite": "S2_act", "scenario": "S2", "seed": 0, "alt": "eval"})
    by_id = {t["id"]: t for t in tasks}
    for s in eval_states:
        if not s["retained_pairs"] or s["split"] not in splits:
            continue
        t = by_id[s["task_id"]]
        naive_rest = t["turn_gt"][s["m"] + 1:] if s["kind"] == "RP" else s["repair_plan"][1:]
        specs.append({"task_id": s["task_id"], "pool": "v2", "alt": "train", "suite": s["kind"], "scenario": "S0", "seed": 0,
                      "forced": s["forced"], "repair_plan": s["repair_plan"], "naive_rest": naive_rest, "state": s["key"]})
    return specs


def load_eval_states(paths):
    out = []
    for p in paths:
        out += [json.loads(l) for l in open(p)]
    return out
