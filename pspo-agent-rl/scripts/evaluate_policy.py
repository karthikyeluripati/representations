"""Metrics with task-clustered bootstrap (cluster = BFCL base task). python scripts/evaluate_policy.py results/x.jsonl [...]
A recoverable completion  : S1/S2 correct (state matches, no escalation)
B correct escalation      : S3 escalated
C false abstention        : escalated on S0/S1/S2
D paired solvability acc. : for each (task, seed) both (S1,S3) and (S2,S3) correct -> averaged over the two pair types
E timeliness              : calls to the faulted function or its backup before escalating on S3 (escalated episodes)
F fault-free success      : S0 correct"""
import collections, json, sys
import numpy as np


def load(p):
    return [json.loads(l) for l in open(p)]


def metrics(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["task_id"], r["seed"])][r["scenario"]] = r
    out = {}
    sc = lambda s: [r for r in rows if r["scenario"] == s]
    rate = lambda xs, k: float(np.mean([bool(x[k]) for x in xs])) if xs else None
    out["fault_free_success"] = rate(sc("S0"), "correct")
    out["recoverable_completion"] = rate(sc("S1") + sc("S2"), "correct")
    out["S1_completion"] = rate(sc("S1"), "correct"); out["S2_completion"] = rate(sc("S2"), "correct")
    out["correct_escalation"] = rate(sc("S3"), "correct")
    out["false_abstention"] = rate(sc("S0") + sc("S1") + sc("S2"), "escalated")
    pairs = []
    for k, d in by.items():
        for a in ("S1", "S2"):
            if a in d and "S3" in d:
                pairs.append(d[a]["correct"] and d["S3"]["correct"])
    out["paired_solvability_acc"] = float(np.mean(pairs)) if pairs else None
    esc = [r["fault_fn_calls"] for r in sc("S3") if r["escalated"]]
    out["timeliness_fault_calls_before_escalation"] = float(np.mean(esc)) if esc else None
    out["mean_tool_calls"] = float(np.mean([r["n_calls"] for r in rows]))
    out["tokens_per_episode"] = float(np.mean([r["tok_in"] + r["tok_out"] for r in rows]))
    out["n_episodes"] = len(rows)
    return out


def boot_ci(rows, key, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["base_id"]].append(r)
    ks = list(groups)
    vals = []
    for _ in range(B):
        pick = rng.choice(len(ks), len(ks), replace=True)
        v = metrics([r for i in pick for r in groups[ks[i]]])[key]
        if v is not None:
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rows = load(p)
        for split in sorted({r["split"] for r in rows}):
            rs = [r for r in rows if r["split"] == split]
            m = metrics(rs)
            ci = boot_ci(rs, "paired_solvability_acc", B=300)
            print(p.split("/")[-1], split, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, "PSA_CI", ci)
