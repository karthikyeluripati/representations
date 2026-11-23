"""Stage-2 descriptive diagnostics (NOT gates; the decision comes from cf_pilot_analysis.py --stage final).
(1) E2 split by compensation present/absent, plus what a contract-blind ceiling allows;
(2) behaviour in compensation-absent, non-decisive timeout cells (where only the contract separates idem from nonidem);
(3) per-seed regression numbers behind S4.
Usage: python scripts/cf_pilot_stage2_diagnostics.py ARCHIVE_ROOT/cf_pilot OUT.json"""
import collections
import glob
import json
import os
import sys

import _common  # noqa: F401
import audit_caw2 as A
from verification import semantic_cf_t2 as T2

R, OUT = sys.argv[1], sys.argv[2]
RUNS = ["CF_s0", "CF_s1", "CF_s2", "ABL_s0", "ABL_s1", "CTRL_s0", "CTRL_s1", "CTRL_s2", "CAW2_s0", "CAW2_s1"]
rows = {n: [json.loads(l) for l in open(f"{R}/t2test/{n}@P0.jsonl", encoding="utf-8")] for n in RUNS}
res = {"E2_by_compensation": {}, "comp_absent_nondecisive_timeouts": {}, "regression_per_seed": {}}
for n, rs in rows.items():
    # E2 definition (semantic_cf_t2.metrics): A rows = non-decisive verifier, groups of 6 (3 effects x 2 outcomes)
    g = collections.defaultdict(list)
    for r in rs:
        if r["verifier"] != "decisive" and r["error"].startswith("timeout"):
            g[(r["family"], r["verifier"], r["compensation"], r["name"], r["error"])].append(r["correct"])
    by = collections.defaultdict(list)
    for k, v in g.items():
        assert len(v) == 6, (k, len(v))
        by[k[2]].append(all(v))
    res["E2_by_compensation"][n] = {c: round(sum(v) / len(v), 3) for c, v in sorted(by.items())}
    res["E2_by_compensation"][n]["pooled"] = round(sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values()), 3)
    c = collections.defaultdict(collections.Counter)
    for r in rs:
        if r["verifier"] != "decisive" and r["compensation"] == "absent" and r["error"].startswith("timeout"):
            c[r["effect"]][r["first"] + ("+" + r["after_check"] if r["after_check"] else "")] += 1
    res["comp_absent_nondecisive_timeouts"][n] = {e: dict(v) for e, v in sorted(c.items())}
feas = {r["task_id"]: r for r in json.load(open(os.path.join(_common.PROJ, "results", "feasibility_ambiguous_commit.json"), encoding="utf-8"))}
ann = json.load(open(os.path.join(_common.PROJ, "results", "tool_annotations.json"), encoding="utf-8"))
ref = {r["task_id"] for r in A.jl(f"{R}/reg/uo/piB.jsonl") if r["suite"] == "S0" and r["correct"]}
ts = {t for t, f in feas.items() if f["decisive"] and f["fault_class"] == "TravelAPI" and t in ref}
import numpy as np  # noqa: E402
for n in ("CF_s0", "CF_s1", "CF_s2", "CTRL_s0", "CAW2_s0", "CAW2_s1"):
    e = {}
    ev = A.jl(f"{R}/reg/eval/{n}.jsonl")
    for sp in ("test", "heldout"):
        e[f"S0_{sp}"] = round(float(np.mean([r["correct"] for r in ev if r["split"] == sp and r["suite"] == "S0"])), 3)
    e["bfcl_heldout_implicit"] = A.bfcl_metrics([A.bfcl_rows(A.jl(f"{R}/reg/uo/{n}.jsonl"), feas, ann)], ts, ("implicit",))[0]["success"]
    m = A.tau2_metrics([A.tau2_rows(A.jl(f"{R}/reg/tau2/{n}.jsonl"))], "diagnostic", ("implicit",), "event")[0]
    e["tau2_check"], e["tau2_success"] = m["check"], m["success"]
    res["regression_per_seed"][n] = e
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(res, indent=1))
