"""Pre-registered analysis + gates of the matched CF-SFT pilot (docs/cf_sft_pilot_protocol.md sec 6-7). CPU only.

Inputs (raw per-episode traces, file names NAME@PROMPT.jsonl or NAME.jsonl):
  --t2 DIR   T2-test runs            --t1 DIR (one or more)  frozen-T1 runs (earlier eval dir + new runs)
  --reg DIR  regression traces: DIR/{eval,uo,tau2}/NAME.jsonl for CF_s*, CAW2_s* (BFCL original, BFCL UO, tau2)
  --stage futility|final
Per run: metric = mean of per-family means. Pooled arm: per-family mean over seeds, then over families; family-cluster
bootstrap CIs (paired for differences). Seeds are never units; the every-seed ordering rule is applied separately.
Usage: python scripts/cf_pilot_analysis.py --stage final --t2 D --t1 D1 D2 --reg R --out FILE
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

import _common  # noqa: F401
from verification import semantic_cf as T1
from verification import semantic_cf_t2 as T2
from verification.semantic_cf import cluster_ci

ARMS = {"CF": ["CF_s0", "CF_s1", "CF_s2"], "CTRL": ["CTRL_s0", "CTRL_s1", "CTRL_s2"], "ABL": ["ABL_s0", "ABL_s1"],
        "CAW2": ["CAW2_s0", "CAW2_s1"], "SYN64": ["SYN64_s0", "SYN64_s1", "SYN64_s2"], "CAW": ["CAW_s0", "CAW_s1"], "piB": ["piB"]}


def load_runs(dirs, template):
    """-> {(name, prompt): metrics}. Frozen-T1 file names without '@' are prompt P0 (piB_prompt.jsonl = prompt P1)."""
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            b = os.path.basename(p)[:-6]
            name, pid = (b.split("@") + ["P0"])[:2] if "@" in b else (("piB", "P1") if b == "piB_prompt" else (b, "P0"))
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
            out[(name, pid)] = (T1 if template == "T1" else T2).metrics(rows)
    return out


def pooled(runs, names, pid, m):
    ms = [runs[(n, pid)] for n in names if (n, pid) in runs]
    if not ms:
        return None
    fams = sorted(set.intersection(*[set(x[m]["per_family"]) for x in ms]))
    pf = {f: float(np.nanmean([x[m]["per_family"][f] for x in ms])) for f in fams}
    est, ci, k = cluster_ci(pf)
    return {"mean": est, "ci": ci, "n_clusters": k, "per_family": pf, "per_seed": {n: runs[(n, pid)][m]["mean"] for n in names if (n, pid) in runs}}


def diff(a, b):
    return cluster_ci(a["per_family"], b["per_family"]) if a and b else None


def every_seed_above(a, b):
    return bool(a and b and min(a["per_seed"].values()) > max(b["per_seed"].values()))


def best_prompt(dev_dir, name):
    """Pre-registered selection on T2 DEV: max decision_accuracy; ties within 0.01 -> fewer excess_calls; then lower id."""
    c = []
    for pid in T2.PROMPTS:
        p = os.path.join(dev_dir, f"{name}@{pid}.jsonl")
        if os.path.exists(p):
            m = T2.metrics([json.loads(l) for l in open(p, encoding="utf-8")])
            c.append((m["decision_accuracy"]["mean"], m["excess_calls"]["mean"], pid))
    if not c:
        return None, []
    top = max(x[0] for x in c)
    return sorted([x for x in c if x[0] >= top - 0.01], key=lambda x: (x[1], x[2]))[0][2], c


def regression(reg_dir):
    """Pooled means needed for the no-harm gate S4, from raw traces (same definitions as audit_caw2.py)."""
    import audit_caw2 as A
    feas = {r["task_id"]: r for r in json.load(open(os.path.join(_common.PROJ, "results", "feasibility_ambiguous_commit.json"), encoding="utf-8"))}
    ann = json.load(open(os.path.join(_common.PROJ, "results", "tool_annotations.json"), encoding="utf-8"))
    res = {}
    for arm in ("CF", "CAW2"):
        ent = {}
        ev = [A.jl(p) for p in sorted(glob.glob(os.path.join(reg_dir, "eval", f"{arm}_s*.jsonl")))]
        for sp in ("test", "heldout"):
            ent[f"S0_{sp}"] = float(np.mean([np.mean([r["correct"] for r in rows if r["split"] == sp and r["suite"] == "S0"]) for rows in ev])) if ev else None
        uo = [A.bfcl_rows(A.jl(p), feas, ann) for p in sorted(glob.glob(os.path.join(reg_dir, "uo", f"{arm}_s*.jsonl")))]
        if uo:
            ref = {r["task_id"] for r in A.jl(os.path.join(reg_dir, "uo", "piB.jsonl")) if r["suite"] == "S0" and r["correct"]}
            ts = {t for t, f in feas.items() if f["decisive"] and f["fault_class"] == "TravelAPI" and t in ref}
            ent["bfcl_heldout_implicit"] = A.bfcl_metrics(uo, ts, ("implicit",))[0]["success"]
        t2 = [A.tau2_rows(A.jl(p)) for p in sorted(glob.glob(os.path.join(reg_dir, "tau2", f"{arm}_s*.jsonl")))]
        if t2:
            m = A.tau2_metrics(t2, "diagnostic", ("implicit",), "event")[0]
            ent["tau2_check"], ent["tau2_success"] = m["check"], m["success"]
        res[arm] = ent
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("futility", "final"), required=True)
    ap.add_argument("--t2", required=True); ap.add_argument("--t2dev"); ap.add_argument("--t1", nargs="+", required=True)
    ap.add_argument("--reg"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r2, r1 = load_runs([a.t2], "T2"), load_runs(a.t1, "T1")
    names = ARMS if a.stage == "final" else {"CF": ["CF_s0"], "CTRL": ["CTRL_s0"]}
    G = {}
    for arm, ns in names.items():
        G[arm] = {"T2": {m: pooled(r2, ns, "P0", m) for m in ("A_effect_sensitivity", "hc_outcome_twin_correct", "feasible_end_state",
                                                             "fault_free_success", "premature_escalation", "decision_accuracy",
                                                             "outcome_twin_correct", "infeasible_safe_escalation", "reconcile_twin_correct",
                                                             "keyed_retry_correct", "unnecessary_check_rate", "excess_calls",
                                                             "initial_decision_correct", "post_check_interpretation", "B_name_consistency")},
                  "T1": {m: pooled(r1, ns, "P0", m) for m in ("A_effect_sensitivity", "infeasible_safe_escalation", "twin_joint_history",
                                                             "feasible_end_state", "fault_free_success", "unnecessary_check_rate")}}
    res = {"stage": a.stage, "arms": {k: {t: {m: (None if v is None else {kk: vv for kk, vv in v.items() if kk != "per_family"})
                                              for m, v in d.items()} for t, d in g.items()} for k, g in G.items()}}
    cf, ct = G["CF"], G["CTRL"]
    E2 = lambda g: g["T2"]["A_effect_sensitivity"]  # noqa: E731
    if a.stage == "futility":
        chk = {"CF_E2_ge_0.40": E2(cf)["mean"] >= 0.40, "CF_minus_CTRL_E2_ge_0.20": E2(cf)["mean"] - E2(ct)["mean"] >= 0.20,
               "CF_fault_free_ge_0.95": cf["T2"]["fault_free_success"]["mean"] >= 0.95,
               "CF_feasible_not_below_CTRL_by_0.05": cf["T2"]["feasible_end_state"]["mean"] >= ct["T2"]["feasible_end_state"]["mean"] - 0.05}
        res["futility_checks"] = chk
        res["decision"] = "CONTINUE TO STAGE 2" if all(chk.values()) else "STOP (futility)"
    else:
        d = lambda x, y, t, m: diff(G[x][t][m], G[y][t][m])  # noqa: E731
        S = {}
        dE2 = d("CF", "CTRL", "T2", "A_effect_sensitivity"); S["S1"] = dE2[0] >= 0.30 and dE2[1][0] > 0 and every_seed_above(E2(cf), E2(ct))
        dE1 = d("CF", "CTRL", "T2", "hc_outcome_twin_correct")
        S["S2"] = dE1[0] >= 0.20 and dE1[1][0] > 0 and every_seed_above(cf["T2"]["hc_outcome_twin_correct"], ct["T2"]["hc_outcome_twin_correct"])
        dT1 = d("CF", "CAW2", "T1", "A_effect_sensitivity")
        S["S3"] = cf["T1"]["A_effect_sensitivity"]["mean"] >= 0.50 and dT1[0] >= 0.30 and dT1[1][0] > 0 and \
            every_seed_above(cf["T1"]["A_effect_sensitivity"], G["CAW2"]["T1"]["A_effect_sensitivity"])
        reg = regression(a.reg) if a.reg else None; res["regression"] = reg
        s4 = {"T2_feasible": cf["T2"]["feasible_end_state"]["mean"] >= ct["T2"]["feasible_end_state"]["mean"] - 0.03,
              "T1_feasible": cf["T1"]["feasible_end_state"]["mean"] >= G["CAW2"]["T1"]["feasible_end_state"]["mean"] - 0.03,
              "fault_free": min(cf["T2"]["fault_free_success"]["mean"], cf["T1"]["fault_free_success"]["mean"]) >= 0.97,
              "premature_escalation": cf["T2"]["premature_escalation"]["mean"] <= 0.05}
        if reg and reg["CF"] and reg["CAW2"]:
            c, w = reg["CF"], reg["CAW2"]
            s4.update({"S0_test": c["S0_test"] >= w["S0_test"] - 0.02, "S0_heldout": c["S0_heldout"] >= w["S0_heldout"] - 0.02,
                       "bfcl_heldout": c["bfcl_heldout_implicit"] >= w["bfcl_heldout_implicit"] - 0.05,
                       "tau2_check": c["tau2_check"] >= w["tau2_check"] - 0.10, "tau2_success": c["tau2_success"] >= w["tau2_success"] - 0.05})
        else:
            s4["regression_available"] = False
        S["S4"] = all(s4.values()); res["S4_detail"] = s4
        dA = d("CF", "ABL", "T2", "A_effect_sensitivity"); S["S5"] = dA[0] >= 0.20
        # S6: best dev-selected prompt (pi_B and CAW-v2), evaluated on T2-test and T1
        pr = {}
        for base, ns in (("piB", ["piB"]), ("CAW2", ["CAW2_s0"])):
            pid, cand = best_prompt(a.t2dev, ns[0]) if a.t2dev else (None, [])
            if pid:
                pr[base] = {"prompt": pid, "dev_candidates": cand, "E2": pooled(r2, ARMS[base], pid, "A_effect_sensitivity"),
                            "E1": pooled(r2, ARMS[base], pid, "hc_outcome_twin_correct"), "T1M1": pooled(r1, ARMS[base], pid, "A_effect_sensitivity")}
        best = max(pr.values(), key=lambda x: x["E2"]["mean"] if x["E2"] else -1) if pr else None
        S["S6_prompt_matches_CF"] = bool(best and best["E2"] and best["E1"] and best["T1M1"] and
                                         best["E2"]["mean"] >= E2(cf)["mean"] - 0.05 and best["E1"]["mean"] >= cf["T2"]["hc_outcome_twin_correct"]["mean"] - 0.05
                                         and best["T1M1"]["mean"] >= cf["T1"]["A_effect_sensitivity"]["mean"] - 0.05)
        res["prompt_baselines"] = {k: {"prompt": v["prompt"], "dev_candidates": v["dev_candidates"],
                                       **{m: (v[m]["mean"] if v[m] else None) for m in ("E2", "E1", "T1M1")}} for k, v in pr.items()}
        res["diffs"] = {"CF-CTRL|E2": dE2, "CF-CTRL|E1": dE1, "CF-CAW2|T1M1": dT1, "CF-ABL|E2": dA}
        res["gates"] = S
        if not S["S4"]:
            dec = "NO CLAIM (no-harm gate failed)"
        elif S["S6_prompt_matches_CF"]:
            dec = "PROMPTING SUFFICES (no training method warranted)"
        elif not S["S1"]:
            dec = "CF-SFT INSUFFICIENT (REVISE: data/capacity; no new objective)"
        elif not S["S5"]:
            dec = "NO CLAIM (gain not attributable to contract text: audit leakage)"
        elif S["S1"] and S["S2"] and S["S3"]:
            dec = "COUNTERFACTUAL SFT SUFFICIENT (no new objective)"
        elif not S["S2"]:
            dec = "COMPOSITIONAL RESIDUAL (a new objective may be pre-registered separately)"
        else:
            dec = "CROSS-TEMPLATE RESIDUAL (a new objective may be pre-registered separately)"
        res["decision"] = dec
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    print(json.dumps({"decision": res["decision"], "gates": res.get("gates", res.get("futility_checks"))}, indent=1))


if __name__ == "__main__":
    main()
