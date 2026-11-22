"""Independent re-computation + decomposition of the CAW-v2 findings from archived raw traces (docs/caw_v2_audit.md).

CPU only. Reads the extracted result archives (pod_backups/*.tgz) and recomputes, per arm and seed:
  tau2:  first post-fault action, check rate, fault-local decision correctness, twin discrimination, end-state success,
         cost; event- AND task-cluster bootstrap (seeds averaged inside a cluster; per-seed values reported separately).
  BFCL:  the uncertain-outcome suites (diagnostic 116 tasks) with the same decomposition; original test/held-out suites.
Fault-local decision (independent of downstream competence):
  ACa (write applied):  correct iff the faulted write is NOT re-issued with identical arguments afterwards
  ACl (write lost):     correct iff the faulted write IS re-issued (same function) afterwards, or its backup endpoint
  twin-discriminating:  both members of an (event, wording, seed) twin pair are decision-correct.
A policy that ignores the verifier's observation cannot exceed 0.5 twin discrimination, whatever its check rate.
Usage: python scripts/audit_caw2.py --root <dir with extracted archives> --out audit.json --supp-out audit_supp.json
(each archive extracted into <root>/<archive name without .tgz>/; replication_slim_20261119.tgz into <root>/rep/)
"""
import argparse
import collections
import json
import os

import numpy as np

ESC = ("escalate_to_user", "transfer_to_human_agents")
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(HERE))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def first_action(calls, k, target, reads):
    if k >= len(calls):
        return "stop"
    n = calls[k][0]
    return "retry" if n == target else "escalate" if n in ESC else "check" if n in reads else "other"


def decision(scn, calls, k, target, targs, alt=None):
    post = calls[k:]
    if scn.startswith("ACa"):
        return not any(n == target and a == targs for n, a, *_ in post)
    return any(n == target or (alt and n == alt) for n, *_ in post)


# ---------------------------------------------------------------- tau2
def tau2_rows(rows):
    out = []
    for r in rows:
        if r["scenario"] == "S0":
            continue
        k = r["n_forced"]; c = r["calls"]; tgt, targs = c[k - 1][0], c[k - 1][1]
        reads = {n for n, *_ in c if n.startswith("get_")} | {n for n, *_ in c[k:] if n.startswith(("get_", "search_", "list_"))}
        fa = first_action(c, k, tgt, reads)
        out.append({"kind": r["kind"], "ev": (r["task"], r["i"]), "task": r["task"], "fn": r["fn"], "scn": r["scenario"], "msg": r["msg"],
                    "first": fa, "check": fa == "check", "dec": decision(r["scenario"], c, k, tgt, targs), "ok": r["correct"],
                    "esc": r["escalated"], "calls": r["n_calls"]})
    return out


def tau2_s0(rows):
    return {(r["task"], r["i"]): r["correct"] for r in rows if r["scenario"] == "S0"}


def cluster_boot(vals_by_cluster_a, vals_by_cluster_b=None, B=4000, seed=0):
    """vals: dict cluster -> value (already averaged over seeds and repeated observations within the cluster)."""
    keys = sorted(vals_by_cluster_a)
    a = np.array([vals_by_cluster_a[k] for k in keys])
    b = np.array([vals_by_cluster_b[k] for k in keys]) if vals_by_cluster_b is not None else np.zeros(len(keys))
    d = a - b; rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), (B, len(keys)))
    m = d[idx].mean(1)
    return [round(float(d.mean()), 3), [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)], len(keys)]


def tau2_metrics(seeds_rows, kind, msgs, cluster):
    """seeds_rows: list (per seed) of tau2_rows output. Returns pooled + per-seed metrics and per-cluster values."""
    per_seed, clus = [], collections.defaultdict(lambda: collections.defaultdict(list))
    for rows in seeds_rows:
        rs = [r for r in rows if r["kind"] == kind and r["msg"] in msgs]
        twins = collections.defaultdict(dict)
        for r in rs:
            twins[(r["ev"], r["msg"])][r["scn"]] = r
        disc = [int(t["ACa"]["dec"] and t["ACl"]["dec"]) for t in twins.values() if len(t) == 2]
        joint = [int(t["ACa"]["ok"] and t["ACl"]["ok"]) for t in twins.values() if len(t) == 2]
        chk_dec = [r["dec"] for r in rs if r["check"]]
        per_seed.append({"n": len(rs), "check": np.mean([r["check"] for r in rs]),
                         "first": dict(collections.Counter(r["first"] for r in rs)),
                         "dec_ACa": np.mean([r["dec"] for r in rs if r["scn"] == "ACa"]),
                         "dec_ACl": np.mean([r["dec"] for r in rs if r["scn"] == "ACl"]),
                         "dec_given_check": np.mean(chk_dec) if chk_dec else float("nan"),
                         "twin_discrim": np.mean(disc), "success": np.mean([r["ok"] for r in rs]), "twin_joint_success": np.mean(joint),
                         "escalate": np.mean([r["esc"] for r in rs]), "calls": np.mean([r["calls"] for r in rs])})
        for key, t in twins.items():
            cl = key[0] if cluster == "event" else key[0][0]
            if len(t) == 2:
                clus[cl]["check"] += [t["ACa"]["check"], t["ACl"]["check"]]
                clus[cl]["disc"].append(int(t["ACa"]["dec"] and t["ACl"]["dec"]))
                clus[cl]["success"] += [t["ACa"]["ok"], t["ACl"]["ok"]]
                clus[cl]["calls"] += [t["ACa"]["calls"], t["ACl"]["calls"]]
    pooled = {k: round(float(np.nanmean([s[k] for s in per_seed])), 3) for k in per_seed[0] if k not in ("first", "n")}
    pooled["per_seed"] = [{k: (round(float(v), 3) if not isinstance(v, (dict, int)) else v) for k, v in s.items()} for s in per_seed]
    cv = {m: {c: float(np.mean(v[m])) for c, v in clus.items()} for m in ("check", "disc", "success", "calls")}
    return pooled, cv


# ---------------------------------------------------------------- BFCL uncertain-outcome suites
def bfcl_rows(rows, feas, ann):
    out = []
    for r in rows:
        if not r["suite"].startswith(("ACa", "ACl")) or r.get("prompt", "std") != "std":
            continue
        f = feas[r["task_id"]]; c = r["calls"]
        # the model issues the faulted call itself (n_forced counts forced prefix only): locate the first call of fault_fn
        ks = [j for j, x in enumerate(c) if x[0] == f["fault_fn"]]
        if not ks:
            out.append({"task": r["task_id"], "suite": r["suite"], "scn": r["suite"][:3], "msg": r["suite"][4:], "first": "never_called",
                        "check": False, "dec": False, "ok": r["correct"], "calls": r["n_calls"], "cls": f["fault_class"], "split": f["split"]})
            continue
        j = ks[0]; tgt, targs = c[j][0], c[j][1]
        reads = set(f["verifiers"]) | {n for n, v in ann.items() if v["readOnlyHint"]}
        fa = first_action(c, j + 1, tgt, reads)
        out.append({"task": r["task_id"], "suite": r["suite"], "scn": r["suite"][:3], "msg": r["suite"][4:], "first": fa,
                    "check": fa == "check", "dec": decision(r["suite"][:3], c, j + 1, tgt, targs), "ok": r["correct"],
                    "calls": r["n_calls"], "cls": f["fault_class"], "split": f["split"]})
    return out


def bfcl_metrics(seeds_rows, tasks, msgs):
    per_seed, clus = [], collections.defaultdict(lambda: collections.defaultdict(list))
    for rows in seeds_rows:
        rs = [r for r in rows if r["task"] in tasks and r["msg"] in msgs]
        tw = collections.defaultdict(dict)
        for r in rs:
            tw[(r["task"], r["msg"])][r["scn"]] = r
        pairs = [t for t in tw.values() if len(t) == 2]
        per_seed.append({"n": len(rs), "check": np.mean([r["check"] for r in rs]), "first": dict(collections.Counter(r["first"] for r in rs)),
                         "dec_ACa": np.mean([r["dec"] for r in rs if r["scn"] == "ACa"]), "dec_ACl": np.mean([r["dec"] for r in rs if r["scn"] == "ACl"]),
                         "twin_discrim": np.mean([t["ACa"]["dec"] and t["ACl"]["dec"] for t in pairs]),
                         "success": np.mean([r["ok"] for r in rs]), "twin_joint_success": np.mean([t["ACa"]["ok"] and t["ACl"]["ok"] for t in pairs]),
                         "success_ACa": np.mean([r["ok"] for r in rs if r["scn"] == "ACa"]), "success_ACl": np.mean([r["ok"] for r in rs if r["scn"] == "ACl"]),
                         "calls": np.mean([r["calls"] for r in rs])})
        for (t, m), p in tw.items():
            if len(p) == 2:
                clus[t]["success"] += [p["ACa"]["ok"], p["ACl"]["ok"]]; clus[t]["disc"].append(int(p["ACa"]["dec"] and p["ACl"]["dec"]))
                clus[t]["check"] += [p["ACa"]["check"], p["ACl"]["check"]]
    pooled = {k: round(float(np.nanmean([s[k] for s in per_seed])), 3) for k in per_seed[0] if k not in ("first", "n")}
    pooled["per_seed"] = [{k: (round(float(v), 3) if not isinstance(v, (dict, int)) else v) for k, v in s.items()} for s in per_seed]
    return pooled, {m: {c: float(np.mean(v[m])) for c, v in clus.items()} for m in ("check", "disc", "success")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--supp-out", help="also write the supplementary analyses (observation sensitivity, fault-free S0)")
    a = ap.parse_args(); R = a.root
    if a.supp_out:
        supp(R, a.supp_out)
    feas = {r["task_id"]: r for r in jl_json(os.path.join(PROJ, "results", "feasibility_ambiguous_commit.json"))}
    ann = jl_json(os.path.join(PROJ, "results", "tool_annotations.json"))
    T2 = {"piB": [f"{R}/tau2_eval_slim_20261121/tau2_eval/piB.jsonl"],
          "SYN32": [f"{R}/tau2_eval_slim_20261121/tau2_eval/SYNIAR_s{s}.jsonl" for s in range(3)],
          "SYN64": [f"{R}/tau2_eval_slim_20261121/tau2_eval/SYN64IAR_s{s}.jsonl" for s in range(3)],
          "CAW": [f"{R}/caw_2seeds_slim_20261121/tau2_eval/CAW64_s{s}.jsonl" for s in range(2)],
          "CAW2": [f"{R}/caw2_2seeds_slim_20261121/tau2_eval/CAW2_s{s}.jsonl" for s in range(2)]}
    res = {"tau2": {}, "bfcl_uo": {}, "original": {}, "notes": []}
    raw = {arm: [jl(p) for p in ps] for arm, ps in T2.items()}
    t2 = {arm: [tau2_rows(x) for x in v] for arm, v in raw.items()}
    res["tau2"]["S0"] = {arm: [round(float(np.mean(list(tau2_s0(x).values()))), 3) for x in v] for arm, v in raw.items()}
    for kind, msgs, name in (("diagnostic", ("implicit",), "diag_implicit"), ("diagnostic", ("explicit", "para", "implicit"), "diag_all"),
                             ("control", ("implicit",), "control_implicit")):
        block = {}; cvs = {}
        for cluster in ("event", "task"):
            for arm in t2:
                p, cv = tau2_metrics(t2[arm], kind, msgs, cluster)
                block[arm] = p; cvs[(arm, cluster)] = cv
        diffs = {}
        for cluster in ("event", "task"):
            for arm in ("CAW2", "CAW"):
                for ref in ("SYN64", "CAW", "piB"):
                    if arm == ref:
                        continue
                    for m in ("check", "disc", "success", "calls"):
                        diffs[f"{arm}-{ref}|{m}|{cluster}"] = cluster_boot(cvs[(arm, cluster)][m], cvs[(ref, cluster)][m])
        res["tau2"][name] = {"arms": block, "diffs": diffs}
    # function breakdown (implicit, diagnostic): count of checked episodes per fn, all seeds summed
    res["tau2"]["by_fn_implicit"] = {arm: {fn: f"{sum(r['check'] for rows in v for r in rows if r['kind']=='diagnostic' and r['msg']=='implicit' and r['fn']==fn)}"
                                                f"/{sum(1 for rows in v for r in rows if r['kind']=='diagnostic' and r['msg']=='implicit' and r['fn']==fn)}"
                                           for fn in ("cancel_reservation", "book_reservation")} for arm, v in t2.items()}
    # --------------------------------------------------- BFCL uncertain-outcome suites
    UO = {"piB": [f"{R}/synth_3seeds_slim_20261121/synth/uo/piB.jsonl"],
          "SYN32": [f"{R}/synth_3seeds_slim_20261121/synth/uo/SYNIAR_s{s}.jsonl" for s in range(3)],
          "CAW": [f"{R}/caw_2seeds_slim_20261121/synth/uo/CAW64_s{s}.jsonl" for s in range(2)],
          "CAW2": [f"{R}/caw2_2seeds_slim_20261121/synth/uo/CAW2_s{s}.jsonl" for s in range(2)]}
    uo_raw = {arm: [jl(p) for p in ps] for arm, ps in UO.items()}
    ref = {r["task_id"]: r for r in uo_raw["piB"][0] if r["suite"] == "S0"}
    TC = ("MessageAPI", "TradingBot", "TwitterAPI", "TicketAPI")
    sets = {"heldout_class_decisive": {t for t, f in feas.items() if f["decisive"] and f["fault_class"] not in TC and t in ref and ref[t]["correct"]},
            "train_class_nontrain_split_decisive": {t for t, f in feas.items() if f["decisive"] and f["fault_class"] in TC and f["split"] != "train"
                                                    and t in ref and ref[t]["correct"]},
            "all_diagnostic": {t for t in ref}}
    uo = {arm: [bfcl_rows(x, feas, ann) for x in v] for arm, v in uo_raw.items()}
    for sname, ts in sets.items():
        for msgs, mn in ((("implicit",), "implicit"), (("para", "implicit"), "unseen_wordings")):
            blk, cvs = {}, {}
            for arm in uo:
                blk[arm], cvs[arm] = bfcl_metrics(uo[arm], ts, msgs)
            diffs = {f"{x}-{y}|{m}": cluster_boot(cvs[x][m], cvs[y][m]) for x in ("CAW2", "CAW") for y in ("CAW", "SYN32", "piB") if x != y
                     for m in ("success", "disc", "check")}
            res["bfcl_uo"][f"{sname}|{mn}"] = {"n_tasks": len(ts), "arms": blk, "diffs": diffs}
    # S1 | S0 (diagnostic 116): standard-retry control
    for arm, v in uo_raw.items():
        out = []
        for rows in v:
            bt = collections.defaultdict(dict)
            for r in rows:
                bt[r["task_id"]][r["suite"]] = r
            ok = [t for t in bt if bt[t]["S0"]["correct"]]
            out.append(round(float(np.mean([bt[t]["S1"]["correct"] for t in ok])), 3))
        res["bfcl_uo"].setdefault("S1_given_S0", {})[arm] = out
    # --------------------------------------------------- original suites (test / heldout), from raw eval traces
    EV = {"B": [f"{R}/rep/replication/eval/B_s{s}.jsonl" for s in range(3)], "B_s0_rerun": [f"{R}/rep/replication/s0_rerun1/B_s0.jsonl"],
          "CAW": [f"{R}/caw_2seeds_slim_20261121/synth/eval/CAW64_s{s}.jsonl" for s in range(2)],
          "CAW2": [f"{R}/caw2_2seeds_slim_20261121/synth/eval/CAW2_s{s}.jsonl" for s in range(2)]}
    for arm, ps in EV.items():
        for p in ps:
            if not os.path.exists(p):
                res["notes"].append(f"missing {os.path.relpath(p, R)}"); continue
            rows = jl(p); s = json.load(open(p.replace(".jsonl", ".summary.json")))
            ent = {}
            for sp in ("test", "heldout"):
                rs = [r for r in rows if r["split"] == sp]
                ent[sp] = {"overall_reported": round(s[sp]["overall"], 3),
                           **{sc: round(float(np.mean([r["correct"] for r in rs if r["suite"] == sc])), 3) for sc in ("S0", "S1", "S2", "S3")}}
            res["original"].setdefault(arm, []).append({os.path.basename(p): ent})
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:200])


def jl_json(p):
    return json.load(open(p, encoding="utf-8"))


# ---------------------------------------------------------------- supplementary analyses (appended; same CLI via --supp)
def obs_sensitivity(rows_by_seed, key_ok=lambda r: True):
    """P(re-issue | checked, ACa) vs P(re-issue | checked, ACl): an observation-blind policy has equal values."""
    out = []
    for rows in rows_by_seed:
        rs = [r for r in rows if r["check"] and key_ok(r)]
        a = [not r["dec"] for r in rs if r["scn"] == "ACa"]; l = [r["dec"] for r in rs if r["scn"] == "ACl"]
        out.append({"n_checked_ACa": len(a), "reissue_after_check_ACa": round(float(np.mean(a)), 3) if a else None,
                    "n_checked_ACl": len(l), "reissue_after_check_ACl": round(float(np.mean(l)), 3) if l else None})
    return out


def s0_paired(eval_by_arm, split, B=4000):
    """Fault-free S0 success, clustered by base task (seeds averaged per task); CAW2 - B and CAW - B."""
    def per_task(rows):
        d = collections.defaultdict(list)
        for r in rows:
            if r["split"] == split and r["suite"] == "S0":
                d[r["base_id"]].append(r["correct"])
        return {k: np.mean(v) for k, v in d.items()}
    pt = {arm: [per_task(x) for x in v] for arm, v in eval_by_arm.items()}
    keys = sorted(set.intersection(*[set(p) for v in pt.values() for p in v]))
    m = {arm: np.array([[p[k] for k in keys] for p in v]).mean(0) for arm, v in pt.items()}
    rng = np.random.default_rng(0); idx = rng.integers(0, len(keys), (B, len(keys))); res = {"n_base_tasks": len(keys)}
    for arm in m:
        res[arm] = round(float(m[arm].mean()), 3)
        if arm != "B":
            d = (m[arm] - m["B"])[idx].mean(1)
            res[f"{arm}-B"] = [round(float((m[arm] - m["B"]).mean()), 3), [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]]
    return res


def supp(R, out):
    feas = {r["task_id"]: r for r in jl_json(os.path.join(PROJ, "results", "feasibility_ambiguous_commit.json"))}
    ann = jl_json(os.path.join(PROJ, "results", "tool_annotations.json"))
    res = {}
    T2 = {"CAW": [f"{R}/caw_2seeds_slim_20261121/tau2_eval/CAW64_s{s}.jsonl" for s in range(2)],
          "CAW2": [f"{R}/caw2_2seeds_slim_20261121/tau2_eval/CAW2_s{s}.jsonl" for s in range(2)]}
    for arm, ps in T2.items():
        rs = [tau2_rows(jl(p)) for p in ps]
        res[f"tau2_obs_sensitivity|{arm}"] = obs_sensitivity(rs, lambda r: r["kind"] == "diagnostic")
    UO = {"SYN32": [f"{R}/synth_3seeds_slim_20261121/synth/uo/SYNIAR_s{s}.jsonl" for s in range(3)],
          "CAW": [f"{R}/caw_2seeds_slim_20261121/synth/uo/CAW64_s{s}.jsonl" for s in range(2)],
          "CAW2": [f"{R}/caw2_2seeds_slim_20261121/synth/uo/CAW2_s{s}.jsonl" for s in range(2)]}
    for arm, ps in UO.items():
        rs = [bfcl_rows(jl(p), feas, ann) for p in ps]
        res[f"bfcl_obs_sensitivity|{arm}|heldout_class"] = obs_sensitivity(rs, lambda r: r["cls"] == "TravelAPI" and feas[r["task"]]["decisive"])
        res[f"bfcl_obs_sensitivity|{arm}|all_diag"] = obs_sensitivity(rs)
    EV = {"B": [f"{R}/rep/replication/eval/B_s{s}.jsonl" for s in range(3)],
          "CAW": [f"{R}/caw_2seeds_slim_20261121/synth/eval/CAW64_s{s}.jsonl" for s in range(2)],
          "CAW2": [f"{R}/caw2_2seeds_slim_20261121/synth/eval/CAW2_s{s}.jsonl" for s in range(2)]}
    ev = {arm: [jl(p) for p in ps] for arm, ps in EV.items()}
    for sp in ("test", "heldout"):
        res[f"S0_fault_free|{sp}"] = s0_paired(ev, sp)
        for arm, ps in EV.items():
            for p in ps:
                s = json.load(open(p.replace(".jsonl", ".summary.json")))[sp]
                res.setdefault(f"composite|{sp}", {})[os.path.basename(p)] = {k: round(s[k], 3) for k in
                    ("fault_free", "injected_recovery", "conditional_recovery", "correct_escalation", "premature_abstention", "false_outage_rate", "overall")}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
