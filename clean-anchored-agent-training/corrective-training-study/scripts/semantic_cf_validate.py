"""CPU validation of the semantic-counterfactual diagnostic (docs/semantic_counterfactual_protocol.md, section 5).

Runs the scripted reference policies on every test-split episode and writes the separation table. The diagnostic is
usable only if (pre-registered): the contract policy scores 1.0 on every metric; check-first scores <= 0.5 on
A_effect_sensitivity; the name heuristic scores <= 0.5 on B_name_consistency's correctness counterpart; and the
always-retry policy scores 0 on infeasible_safe_escalation.
Usage: python scripts/semantic_cf_validate.py [--out data/semantic_cf_v0/validation.json]
"""
import argparse
import hashlib
import json
import os

import _common
from verification.semantic_cf import POLICIES, SYSTEM, metrics, run_policy, specs

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(_common.STUDY, "data", "semantic_cf_v0", "validation.json"))
a = ap.parse_args()
sp = specs("test")
res = {"n_episodes": len(sp), "n_families": len({s["family"] for s in sp}),
       "specs_sha256": hashlib.sha256(json.dumps(sp, sort_keys=True).encode()).hexdigest(), "policies": {}}
keys = ("decision_accuracy", "initial_decision_correct", "post_check_interpretation", "outcome_twin_correct", "feasible_end_state", "infeasible_safe_escalation", "A_effect_sensitivity", "B_name_consistency",
        "B_name_both_correct", "twin_joint_listing", "twin_joint_history", "check_rate_when_required", "unnecessary_check_rate",
        "check_rate_after_fault", "excess_calls", "premature_escalation", "duplicate_rate", "mean_calls", "fault_free_success")
for name, P in POLICIES.items():
    m = metrics(run_policy(P(), sp, SYSTEM))
    res["policies"][name] = {k: m[k]["mean"] for k in keys} | {"cells": m["cells"]}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
w = max(len(k) for k in keys)
print(f"{'':{w}} " + " ".join(f"{n[:12]:>12}" for n in POLICIES))
for k in keys:
    print(f"{k:{w}} " + " ".join(f"{str(res['policies'][n][k]):>12}" for n in POLICIES))
