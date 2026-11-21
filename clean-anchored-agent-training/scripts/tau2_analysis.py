"""Pooled analysis of the tau2 airline transfer test (docs/tau2_transfer.md). Event-cluster bootstrap.
Usage: python scripts/tau2_analysis.py --dir DIR --arms SYNOLD SYNIAR SYN64IAR --seeds 0 1 2 --ref piB --out FILE"""
import argparse
import collections
import json
import os

import numpy as np


def load(d, n):
    return [json.loads(l) for l in open(os.path.join(d, n + ".jsonl"), encoding="utf-8")]


def checked(r):
    c, k = r["calls"], r["n_forced"]
    return int(k < len(c) and c[k][0].startswith("get_"))


ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
ap.add_argument("--ref", default="piB"); ap.add_argument("--B", type=int, default=2000); ap.add_argument("--out", required=True)
ap.add_argument("--base", nargs="+", default=["SYNOLD"], help="arms every other arm is compared against")
a = ap.parse_args()
runs = {arm: [load(a.dir, f"{arm}_s{s}") for s in a.seeds if os.path.exists(os.path.join(a.dir, f"{arm}_s{s}.jsonl"))] for arm in a.arms}
runs[a.ref] = [load(a.dir, a.ref)]
res = {"n_seeds": {k: len(v) for k, v in runs.items()}}
for kind, msgs in (("diagnostic", ("implicit", "para", "explicit")), ("control", ("implicit",))):
    for m in msgs:
        key = f"{kind}|{m}"
        ev = sorted({(r["task"], r["i"]) for r in runs[a.ref][0] if r["kind"] == kind})
        per = {}
        for arm, rs in runs.items():
            chk = np.zeros((len(rs), len(ev))); suc = np.zeros((len(rs), len(ev))); calls = np.zeros((len(rs), len(ev)))
            for si, rows in enumerate(rs):
                by = collections.defaultdict(list)
                for r in rows:
                    if r["kind"] == kind and r["msg"] == m:
                        by[(r["task"], r["i"])].append(r)
                for ei, e in enumerate(ev):
                    xs = by[e]
                    chk[si, ei] = np.mean([checked(r) for r in xs]); suc[si, ei] = np.mean([r["correct"] for r in xs])
                    calls[si, ei] = np.mean([r["n_calls"] for r in xs])
            per[arm] = {"check": chk.mean(0), "success": suc.mean(0), "calls": calls.mean(0),
                        "per_seed_check": chk.mean(1).round(3).tolist(), "per_seed_success": suc.mean(1).round(3).tolist()}
        rng = np.random.default_rng(0); idx = rng.integers(0, len(ev), (a.B, len(ev)))
        out = {arm: {"check": round(float(v["check"].mean()), 3), "success": round(float(v["success"].mean()), 3),
                     "mean_calls": round(float(v["calls"].mean()), 2), "per_seed_check": v["per_seed_check"],
                     "per_seed_success": v["per_seed_success"]} for arm, v in per.items()}
        diffs = {}
        for b in a.base:
            for arm in a.arms:
                if arm == b or arm not in per or b not in per:
                    continue
                for met in ("check", "success"):
                    d = per[arm][met][idx].mean(1) - per[b][met][idx].mean(1)
                    diffs[f"{arm} - {b}|{met}"] = [round(float(per[arm][met].mean() - per[b][met].mean()), 3),
                                                   [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]]
        res[key] = {"n_events": len(ev), "arms": out, "diffs": diffs}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
