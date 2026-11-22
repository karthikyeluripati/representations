"""CPU separation table of scripted policies on T2 dev and test (docs/cf_sft_pilot_protocol.md sec 5).
Usage: python scripts/cf_pilot_validate.py  -> data/cf_sft_v1/validation_t2.json"""
import hashlib
import json
import os

import _common
from verification import semantic_cf_t2 as T

KEYS = ("decision_accuracy", "initial_decision_correct", "post_check_interpretation", "outcome_twin_correct", "feasible_end_state",
        "infeasible_safe_escalation", "reconcile_twin_correct", "keyed_retry_correct", "A_effect_sensitivity", "B_name_consistency",
        "hc_decision_accuracy", "hc_outcome_twin_correct", "premature_escalation", "duplicate_rate", "unnecessary_check_rate",
        "excess_calls", "fault_free_success")
res = {}
for split in ("dev", "test"):
    sp = T.specs(split)
    res[split] = {"n_episodes": len(sp), "n_families": len({s["family"] for s in sp}),
                  "specs_sha256": hashlib.sha256(json.dumps(sp, sort_keys=True).encode()).hexdigest(), "policies": {}}
    for n, P in T.POLICIES_T2.items():
        m = T.metrics(T.run_policy(P(), sp)); res[split]["policies"][n] = {k: m[k]["mean"] for k in KEYS}
json.dump(res, open(os.path.join(_common.STUDY, "data", "cf_sft_v1", "validation_t2.json"), "w", encoding="utf-8", newline="\n"), indent=1)
for split in res:
    print(split, res[split]["n_episodes"], res[split]["specs_sha256"][:8])
    for k in KEYS:
        print(f"  {k:28s}" + "".join(f"{str(res[split]['policies'][n][k]):>10}" for n in T.POLICIES_T2))
