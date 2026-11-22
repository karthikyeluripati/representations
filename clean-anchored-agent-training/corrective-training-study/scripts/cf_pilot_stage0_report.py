"""Stage-0 report of the matched CF-SFT pilot (docs/cf_sft_pilot_protocol.md sec 3, 5): descriptive only, no gate.
Recomputes every metric from raw traces; family-cluster bootstrap CIs (seeds averaged inside a family).
Usage: python scripts/cf_pilot_stage0_report.py --root DIR(with cf_pilot/{t2dev,t2test,t1}) --t1frozen DIR --out FILE.json --md FILE.md"""
import argparse
import glob
import json
import os

import _common  # noqa: F401
from cf_pilot_analysis import best_prompt
from verification import semantic_cf as T1
from verification import semantic_cf_t2 as T2
from verification.semantic_cf import cluster_ci

M2 = ["decision_accuracy", "A_effect_sensitivity", "hc_outcome_twin_correct", "feasible_end_state", "infeasible_safe_escalation",
      "reconcile_twin_correct", "keyed_retry_correct", "outcome_twin_correct", "premature_escalation", "duplicate_rate",
      "unnecessary_check_rate", "excess_calls", "fault_free_success"]
M1 = ["decision_accuracy", "A_effect_sensitivity", "twin_joint_history", "feasible_end_state", "infeasible_safe_escalation",
      "premature_escalation", "duplicate_rate", "unnecessary_check_rate", "excess_calls", "fault_free_success"]
GROUPS = {"piB": ["piB"], "SYN64": ["SYN64_s0", "SYN64_s1", "SYN64_s2"], "CAW": ["CAW_s0", "CAW_s1"], "CAW2": ["CAW2_s0", "CAW2_s1"]}


def load(d, mod):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        out[os.path.basename(p)[:-6]] = (mod.metrics(rows), len(rows))
    return out


def pool(runs, names, metrics):
    have = [n for n in names if n in runs]
    if not have:
        return None
    out = {"runs": have, "n_episodes_per_run": runs[have[0]][1]}
    for m in metrics:
        fams = sorted(set.intersection(*[set(runs[n][0][m]["per_family"]) for n in have]))
        pf = {f: sum(runs[n][0][m]["per_family"][f] for n in have) / len(have) for f in fams}
        est, ci, k = cluster_ci(pf)
        out[m] = {"mean": est, "ci": ci, "n_families": k, "per_seed": {n: runs[n][0][m]["mean"] for n in have}}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--t1frozen", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args(); R = os.path.join(a.root, "cf_pilot")
    dev, test, t1 = load(os.path.join(R, "t2dev"), T2), load(os.path.join(R, "t2test"), T2), load(os.path.join(R, "t1"), T1)
    frozen = {k + "@P0": v for k, v in load(a.t1frozen, T1).items() if k != "piB_prompt"}
    frozen["piB@P1"] = load(a.t1frozen, T1)["piB_prompt"]
    res = {"selection": {}, "dev": {}, "test": {}, "t1": {}}
    for name in ("piB", "CAW2_s0"):
        pid, cand = best_prompt(os.path.join(R, "t2dev"), name)
        res["selection"][name] = {"selected": pid, "candidates_(dev_acc, dev_excess_calls, prompt)": cand}
        for p in T2.PROMPTS:
            res["dev"][f"{name}@{p}"] = pool(dev, [f"{name}@{p}"], M2)
    sel = {"piB": res["selection"]["piB"]["selected"], "CAW2": res["selection"]["CAW2_s0"]["selected"]}
    for g, ns in GROUPS.items():
        for p in sorted({"P0", sel.get(g, "P0")}):
            r = pool(test, [f"{n}@{p}" for n in ns], M2)
            if r:
                res["test"][f"{g}@{p}"] = r
            r1 = pool({**frozen, **t1}, [f"{n}@{p}" for n in ns], M1)
            if r1:
                res["t1"][f"{g}@{p}"] = r1
    for g in ("piB", "CAW2"):
        if sel[g] != "P0":
            A, B = res["test"][f"{g}@{sel[g]}"], res["test"][f"{g}@P0"]
            res["test"][f"{g}@{sel[g]} - {g}@P0"] = {m: cluster_ci(*[{f: sum(test[f'{n}@{p}'][0][m]['per_family'][f] for n in GROUPS[g]) / len(GROUPS[g])
                                                                    for f in test[f'{GROUPS[g][0]}@{p}'][0][m]['per_family']} for p in (sel[g], "P0")])
                                                     for m in ("A_effect_sensitivity", "hc_outcome_twin_correct", "decision_accuracy")}
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    show2 = ["decision_accuracy", "A_effect_sensitivity", "hc_outcome_twin_correct", "feasible_end_state", "infeasible_safe_escalation",
             "reconcile_twin_correct", "unnecessary_check_rate", "excess_calls", "fault_free_success"]
    fmt = lambda v: f"{v['mean']} [{v['ci'][0]}, {v['ci'][1]}]" if v and v["mean"] is not None else "—"  # noqa: E731
    L = ["## T2-dev (prompt selection; 10 families, 2,160 episodes per run)", "| run | " + " | ".join(show2) + " |", "|---" * (len(show2) + 1) + "|"]
    L += [f"| {k} | " + " | ".join(fmt(v[m]) for m in show2) + " |" for k, v in res["dev"].items()]
    L += ["", f"Selected: {res['selection']}", "", "## T2-test (20 families, 4,320 episodes per run; seeds pooled)",
          "| arm@prompt (runs) | " + " | ".join(show2) + " |", "|---" * (len(show2) + 1) + "|"]
    L += [f"| {k} ({len(v['runs'])}) | " + " | ".join(fmt(v[m]) for m in show2) + " |" for k, v in res["test"].items() if " - " not in k]
    show1 = ["decision_accuracy", "A_effect_sensitivity", "twin_joint_history", "feasible_end_state", "infeasible_safe_escalation",
             "unnecessary_check_rate", "excess_calls", "fault_free_success"]
    L += ["", "## T1 frozen test (20 families, 1,920 episodes per run)", "| arm@prompt (runs) | " + " | ".join(show1) + " |", "|---" * (len(show1) + 1) + "|"]
    L += [f"| {k} ({len(v['runs'])}) | " + " | ".join(fmt(v[m]) for m in show1) + " |" for k, v in res["t1"].items()]
    open(a.md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
