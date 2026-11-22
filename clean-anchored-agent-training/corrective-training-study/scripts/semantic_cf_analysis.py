"""Pre-registered analysis of the semantic-counterfactual evaluation (docs/semantic_counterfactual_protocol.md sec 6 +
edit log 1). Applies the decision gates mechanically; nothing here is tuned on results.

Per run (arm, seed): every metric as a mean of per-family means. Pooled arm value: per-family mean over the arm's seeds,
then mean over families. CIs: family-cluster bootstrap (B=4000, seed 0); one-sample for gates, paired for differences.
Seeds are never counted as independent units; per-seed values are reported separately.
Scripted reference policies (CPU) are included as rows.
Usage: python scripts/semantic_cf_analysis.py --dir DIR --out FILE.json [--md FILE.md]
"""
import argparse
import json
import os

import _common  # noqa: F401
from verification.semantic_cf import POLICIES, SYSTEM, cluster_ci, metrics, run_policy, specs

ARMS = {"piB": ["piB"], "piB_prompt": ["piB_prompt"], "SYN64": ["SYN64_s0", "SYN64_s1", "SYN64_s2"],
        "CAW": ["CAW_s0", "CAW_s1"], "CAW2": ["CAW2_s0", "CAW2_s1"]}
METRICS = ["A_effect_sensitivity", "twin_joint_history", "infeasible_safe_escalation", "feasible_end_state",
           "B_name_consistency", "B_name_both_correct", "unnecessary_check_rate", "twin_joint_listing", "fault_free_success",
           "decision_accuracy", "initial_decision_correct", "check_rate_after_fault", "post_check_interpretation",
           "outcome_twin_correct", "premature_escalation", "duplicate_rate", "excess_calls", "mean_calls"]


def per_family(runs, m):
    fams = sorted(set.intersection(*[set(r[m]["per_family"]) for r in runs]))
    out = {}
    for f in fams:
        v = [r[m]["per_family"][f] for r in runs if r[m]["per_family"][f] == r[m]["per_family"][f]]
        out[f] = sum(v) / len(v) if v else float("nan")
    return out


def gates(G, prompt):
    """Protocol sec 6, applied in the order written."""
    comp = G["fault_free_success"]["mean"] >= 0.85 and G["twin_joint_listing"]["mean"] >= 0.6
    M1, M2, M3 = G["A_effect_sensitivity"], G["twin_joint_history"], G["infeasible_safe_escalation"]
    log = {"competence_precondition": comp}
    if not comp:
        return "REVISE HYPOTHESIS (inconclusive: competence)", log
    fals = M1["mean"] >= 0.70 and M1["ci"][0] > 0.5 and M2["mean"] >= 0.70 and M3["mean"] >= 0.70
    supp = M1["ci"][1] < 0.5 or M2["ci"][1] < 0.5
    pclose = prompt["A_effect_sensitivity"]["mean"] >= 0.70 and prompt["twin_joint_history"]["mean"] >= 0.70
    log.update(H_s_falsified=fals, H_s_supported=supp, prompt_closes_gap=pclose)
    if fals:
        return "STOP METHOD PURSUIT (CAW-v2 conditions semantically)", log
    if supp and pclose:
        return "REVISE HYPOTHESIS (gap closable by prompting)", log
    if supp:
        return "PROCEED TO MATCHED SFT PILOT", log
    return "REVISE HYPOTHESIS", log


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--md")
    a = ap.parse_args()
    sp = specs("test"); n_fam = len({s["family"] for s in sp})
    runs = {}
    for arm, names in ARMS.items():
        for n in names:
            p = os.path.join(a.dir, n + ".jsonl")
            if os.path.exists(p):
                rows = [json.loads(l) for l in open(p, encoding="utf-8")]
                assert len(rows) == len(sp), (n, len(rows))
                runs.setdefault(arm, {})[n] = metrics(rows)
    for name, P in POLICIES.items():
        runs[f"scripted:{name}"] = {name: metrics(run_policy(P(), sp, SYSTEM))}
    res = {"n_episodes_per_run": len(sp), "n_family_clusters": n_fam, "arms": {}, "diffs": {}}
    pooled = {}
    for arm, rs in runs.items():
        ent = {"seeds": list(rs), "per_seed": {n: {m: r[m]["mean"] for m in METRICS} for n, r in rs.items()}, "pooled": {}}
        pooled[arm] = {}
        for m in METRICS:
            pf = per_family(list(rs.values()), m); pooled[arm][m] = pf
            est, ci, k = cluster_ci(pf)
            ent["pooled"][m] = {"mean": est, "ci": ci, "n_clusters": k}
        ent["cells"] = {n: r["cells"] for n, r in rs.items()}
        res["arms"][arm] = ent
    for x in ("CAW2", "CAW", "piB_prompt"):
        for y in ("SYN64", "piB", "CAW", "scripted:always_check"):
            if x != y and x in pooled and y in pooled:
                for m in METRICS:
                    res["diffs"][f"{x} - {y}|{m}"] = cluster_ci(pooled[x][m], pooled[y][m])
    if "CAW2" in res["arms"] and "piB_prompt" in res["arms"]:
        res["decision"], res["gate_log"] = gates(res["arms"]["CAW2"]["pooled"], res["arms"]["piB_prompt"]["pooled"])
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    if a.md:
        show = ["initial_decision_correct", "check_rate_after_fault", "post_check_interpretation", "outcome_twin_correct",
                "feasible_end_state", "infeasible_safe_escalation", "A_effect_sensitivity", "twin_joint_history", "B_name_consistency",
                "premature_escalation", "duplicate_rate", "unnecessary_check_rate", "excess_calls", "fault_free_success"]
        L = ["| arm (seed) | " + " | ".join(show) + " |", "|---" * (len(show) + 1) + "|"]
        for arm, e in res["arms"].items():
            L.append(f"| **{arm} pooled** | " + " | ".join(f"{e['pooled'][m]['mean']} [{e['pooled'][m]['ci'][0]}, {e['pooled'][m]['ci'][1]}]" for m in show) + " |")
            if len(e["per_seed"]) > 1:
                for n, v in e["per_seed"].items():
                    L.append(f"| {n} | " + " | ".join(str(v[m]) for m in show) + " |")
        L.append(f"\nDecision (protocol sec 6): **{res.get('decision')}**; gate log: {res.get('gate_log')}")
        open(a.md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(json.dumps({"decision": res.get("decision"), "gate_log": res.get("gate_log"),
                      **{arm: {m: e["pooled"][m]["mean"] for m in METRICS[:6]} for arm, e in res["arms"].items()}}, indent=1))


if __name__ == "__main__":
    main()
