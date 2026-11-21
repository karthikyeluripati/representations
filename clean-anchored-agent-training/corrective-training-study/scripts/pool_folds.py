"""Pool the leave-one-class-out folds of the evidence-grounded study: concatenate each fold's held-out-class tasks
(per-task values from otv_analysis.py --out files) and bootstrap the method - baseline difference over tasks.
Usage: python scripts/pool_folds.py --fold M analysis_M.json EGRM AAC2M --fold T analysis_T.json EGRT AAC2T --out pooled.json"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fold", nargs=4, action="append", required=True, metavar=("NAME", "FILE", "METHOD", "BASE"))
ap.add_argument("--key", default="unseen_class|unseen_wordings"); ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m, b, per = [], [], {}
for name, f, M, Bn in a.fold:
    tv = json.load(open(f))["task_values"][a.key]
    m += tv[M]; b += tv[Bn]
    per[name] = {"n_tasks": len(tv["tasks"]), "method": float(np.nanmean(tv[M])), "base": float(np.nanmean(tv[Bn]))}
m, b = np.array(m), np.array(b)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(m), (a.B, len(m)))
d = np.nanmean(m[idx], 1) - np.nanmean(b[idx], 1)
res = {"key": a.key, "folds": per, "n_tasks": len(m), "method": float(np.nanmean(m)), "base": float(np.nanmean(b)),
       "diff": float(np.nanmean(m) - np.nanmean(b)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
