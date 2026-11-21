"""Pool per-seed evaluations of each arm and compare arms with a seed-pooled, base-task cluster bootstrap.

For each bootstrap draw, BFCL base tasks are resampled with replacement; an arm's metric is computed on the
resampled episodes of EVERY seed and averaged over seeds. A difference arm1 - arm2 uses the same draw, so it is
paired by task and averaged over training seeds (training-seed variance enters through the per-seed means;
per-seed point estimates are reported too).
Usage: python scripts/pool_seeds.py --arm A results/eval/A_s0.jsonl ... --arm B ... --ref results/eval/pi0.jsonl
"""
import argparse
import collections
import json

import numpy as np

import _common  # noqa: F401
from src.evaluation import metrics as M

KEYS = ["overall", "fault_free", "genuine_repair", "injected_recovery", "conditional_recovery", "correct_escalation",
        "premature_abstention", "false_outage_rate"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", required=True, help="NAME file_seed0 file_seed1 ...")
    ap.add_argument("--ref", default=None, help="single-run reference (pi0), treated as an arm with one 'seed'")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {x[0]: [load(f) for f in x[1:]] for x in a.arm}
    if a.ref:
        arms["pi0"] = [load(a.ref)]
    rng = np.random.default_rng(0)
    res = {"per_seed": {}, "pooled": {}, "diffs": {}}
    for sp in ("test", "heldout"):
        A = {n: [[r for r in rows if r["split"] == sp] for rows in runs] for n, runs in arms.items()}
        res["per_seed"][sp] = {n: [{k: M.summarize(rs)[k] for k in KEYS} for rs in runs] for n, runs in A.items()}
        idx = {n: [collections.defaultdict(list) for _ in runs] for n, runs in A.items()}
        for n, runs in A.items():
            for i, rs in enumerate(runs):
                for r in rs:
                    idx[n][i][r["base_id"]].append(r)
        bases = sorted({b for n in idx for d in idx[n] for b in d})
        draws = {n: {k: [] for k in KEYS} for n in A}
        for _ in range(a.B):
            pick = [bases[j] for j in rng.integers(0, len(bases), len(bases))]
            for n in A:
                vals = collections.defaultdict(list)
                for d in idx[n]:
                    s = M.summarize([r for b in pick for r in d.get(b, [])])
                    for k in KEYS:
                        if s[k] is not None:
                            vals[k].append(s[k])
                for k in KEYS:
                    draws[n][k].append(float(np.mean(vals[k])) if vals[k] else np.nan)
        res["pooled"][sp] = {n: {k: float(np.mean([s[k] for s in res["per_seed"][sp][n] if s[k] is not None])) for k in KEYS} for n in A}
        names = list(A)
        res["diffs"][sp] = {}
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                res["diffs"][sp][f"{y} - {x}"] = {
                    k: [res["pooled"][sp][y][k] - res["pooled"][sp][x][k],
                        [float(np.nanpercentile(np.array(draws[y][k]) - np.array(draws[x][k]), q)) for q in (2.5, 97.5)]] for k in KEYS}
    json.dump(res, open(a.out, "w"), indent=1)
    for sp in ("test", "heldout"):
        print(sp, {n: round(v["overall"], 3) for n, v in res["pooled"][sp].items()})
        for d, v in res["diffs"][sp].items():
            print("  ", d, "overall", round(v["overall"][0], 3), [round(c, 3) for c in v["overall"][1]])


if __name__ == "__main__":
    main()
