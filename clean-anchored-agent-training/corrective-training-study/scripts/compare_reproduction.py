"""Pre-registered reproduction check for the rebuilt starting policy pi0.

Compares a rebuilt pi0 evaluation with the ARCHIVED pi0 evaluation on identical episodes:
  * per-split headline metrics and their paired cluster-bootstrap differences;
  * episode-level agreement of the grader outcome.
Rule (fixed before running): the archived pairs are reused only if, on BOTH test and heldout, episode agreement
>= 0.90 AND every headline paired-difference CI contains 0. Otherwise the pairs are regenerated under the rebuilt
pi0 (parent `evaluate.py --mode sample_states`) and the cts datasets rebuilt; both outcomes are logged.
Exit code 0 = reuse, 3 = regenerate.
"""
import argparse
import io
import json
import sys
import tarfile

import _common  # noqa: F401
from src.evaluation import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--rebuilt", required=True)
ap.add_argument("--archive", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
new = [json.loads(l) for l in open(a.rebuilt, encoding="utf-8")]
old = [json.loads(l) for l in io.BytesIO(tarfile.open(a.archive).extractfile("clean-anchored-agent-training/results/eval/pi0.jsonl").read())]
key = lambda r: (r["task_id"], r["suite"], r["seed"], r.get("state"))  # noqa: E731
rep, reuse = {}, True
for sp in ("test", "heldout"):
    A = [r for r in new if r["split"] == sp]; B = [r for r in old if r["split"] == sp]
    kb = {key(r): r["correct"] for r in B}
    agree = sum(kb.get(key(r)) == r["correct"] for r in A) / max(len(A), 1)
    diffs = {k: M.paired_diff(A, B, M.metric_fn(k), B=1000) for k in ("fault_free", "genuine_repair", "false_outage_rate", "overall")}
    ok = agree >= 0.90 and all(lo <= 0 <= hi for _, (lo, hi) in diffs.values())
    reuse &= ok
    rep[sp] = {"episode_agreement": agree, "paired_diff_rebuilt_minus_archived": diffs, "within_tolerance": ok}
rep["decision"] = "reuse_archived_pairs" if reuse else "regenerate_pairs"
json.dump(rep, open(a.out, "w"), indent=1)
print(json.dumps(rep, indent=1))
sys.exit(0 if reuse else 3)
