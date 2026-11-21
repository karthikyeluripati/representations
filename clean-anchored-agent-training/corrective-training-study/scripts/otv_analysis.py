"""Primary analysis of the OTV method study (docs/method_outcome_twins.md, edit log 2).

Fixed task set = decisive unseen-class tasks (not TravelAPI/TradingBot) that pi_B solves fault-free under std.
Per-task primary value = mean success over {ACa, ACl} x {para, implicit} (std prompt). Arms are compared with a
seed-pooled paired bootstrap over tasks. Also reports in-distribution (decisive Travel/Trading tasks outside the train
split), per-wording, S1|S0, and extra calls on S0 vs pi_B.
Usage: python scripts/otv_analysis.py --uo DIR --ref piB --arms OTV BAC AAC --seeds 0 [1 2] --out FILE
"""
import argparse
import collections
import json
import os

import numpy as np

import _common

TRAIN_CLASSES = ("TravelAPI", "TradingBot")
UNSEEN = ("para", "implicit")


def load(d, n):
    return [json.loads(l) for l in open(os.path.join(d, n + ".jsonl"), encoding="utf-8") if json.loads(l)["prompt"] == "std"]


def by_task(rows):
    out = collections.defaultdict(dict)
    for r in rows:
        out[r["task_id"]][r["suite"]] = r
    return out


def task_value(t, wordings):
    xs = [t[f"{v}_{m}"]["correct"] for v in ("ACa", "ACl") for m in wordings if f"{v}_{m}" in t]
    return float(np.mean(xs)) if xs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uo", required=True); ap.add_argument("--ref", default="piB")
    ap.add_argument("--arms", nargs="+", default=["OTV", "BAC", "AAC"]); ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--out", required=True)
    ap.add_argument("--train-classes", default=",".join(TRAIN_CLASSES)); ap.add_argument("--method", default="OTV")
    a = ap.parse_args()
    TC = tuple(a.train_classes.split(","))
    feas = {r["task_id"]: r for r in json.load(open(os.path.join(_common.PROJ, "results", "feasibility_ambiguous_commit.json"), encoding="utf-8"))}
    ref = by_task(load(a.uo, a.ref))
    groups = {
        "unseen_class": [t for t, r in feas.items() if r["decisive"] and r["fault_class"] not in TC],
        "in_dist": [t for t, r in feas.items() if r["decisive"] and r["fault_class"] in TC and r["split"] != "train"]}
    fixed = {g: sorted(t for t in ts if t in ref and ref[t]["S0"]["correct"]) for g, ts in groups.items()}
    runs = {arm: {s: by_task(load(a.uo, f"{arm}_s{s}")) for s in a.seeds} for arm in a.arms}
    runs[a.ref] = {"-": ref}
    res = {"n_tasks": {g: len(v) for g, v in fixed.items()}, "per_seed": {}, "pooled": {}, "diffs": {}, "controls": {}}
    rng = np.random.default_rng(0)
    for g, ts in fixed.items():
        for wset, wn in ((UNSEEN, "unseen_wordings"), (("explicit",), "train_wording"), (("para",), "para"), (("implicit",), "implicit")):
            key = f"{g}|{wn}"
            vals = {arm: np.array([[task_value(runs[arm][s][t], wset) for t in ts] for s in runs[arm]]) for arm in runs}
            res["per_seed"][key] = {arm: [round(float(np.nanmean(v[i])), 4) for i in range(len(v))] for arm, v in vals.items()}
            res["pooled"][key] = {arm: round(float(np.nanmean(v)), 4) for arm, v in vals.items()}
            res.setdefault("task_values", {})[key] = {"tasks": ts, **{arm: v.mean(0).tolist() for arm, v in vals.items()}}
            if ts:
                idx = rng.integers(0, len(ts), (a.B, len(ts)))
                draws = {arm: np.nanmean(v.mean(0)[idx], axis=1) for arm, v in vals.items()}
                res["diffs"][key] = {f"{x} - {y}": [round(float(np.mean(draws[x] - draws[y])), 4),
                                                    [round(float(np.percentile(draws[x] - draws[y], q)), 4) for q in (2.5, 97.5)]]
                                     for x in a.arms for y in list(a.arms) + [a.ref] if x != y}
    # controls on all 116 diagnostic tasks: S1 given S0, extra calls on S0 vs ref
    for arm in runs:
        for s, bt in runs[arm].items():
            s0ok = [t for t in bt if bt[t]["S0"]["correct"]]
            s1 = np.mean([bt[t]["S1"]["correct"] for t in s0ok]) if s0ok else np.nan
            extra = np.mean([bt[t]["S0"]["n_calls"] - ref[t]["S0"]["n_calls"] for t in bt if t in ref])
            res["controls"][f"{arm}|{s}"] = {"S0": round(float(np.mean([bt[t]["S0"]["correct"] for t in bt])), 4),
                                            "S1_given_S0": round(float(s1), 4), "extra_calls_S0": round(float(extra), 3)}
    prim = "unseen_class|unseen_wordings"
    M = a.method
    base = {arm: res["pooled"][prim][arm] for arm in a.arms if arm != M}
    best = max(base, key=base.get) if base else None
    res["primary"] = {"key": prim, "method": M, "best_baseline": best,
                      "method_minus_best": res["diffs"].get(prim, {}).get(f"{M} - {best}") if best else None,
                      "seed0_margin": (res["per_seed"][prim][M][0] - max(res["per_seed"][prim][b][0] for b in base)) if base else None}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"n_tasks": res["n_tasks"], "pooled_primary": res["pooled"][prim], "primary": res["primary"],
                      "in_dist": res["pooled"]["in_dist|unseen_wordings"], "controls": res["controls"]}, indent=1))


if __name__ == "__main__":
    main()
