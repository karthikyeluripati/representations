"""Metrics with base-task cluster bootstrap (rollouts / seeds / turns of one BFCL base task are NOT independent).

  success[suite]               P(correct) per suite
  fault_free                   = success[S0]
  conditional_recovery         P(correct | S1/S2 episode of a task whose S0 episode is correct)   (same policy)
  genuine_repair               success over RP+RA
  injected_recovery            success over S1+S2 (unconditional)
  correct_escalation           success[S3]
  premature_abstention         P(escalated) over solvable episodes (S0, S1*, S2*, RP, RA)
  false_outage_rate            over REAL-error observations caused by the agent's own (or forced) calls: share whose
                               next agent action is an outage response (identical retry, alternative endpoint, escalate)
  overall                      mean of the per-suite success rates (macro; every suite weighted equally)
  retention(pi, ref)           P(pi correct | ref correct) on S0, matched tasks
"""
from __future__ import annotations

import collections

import numpy as np

from src.environment.bfcl_env import ESCALATE, obs_kind

SOLVABLE = {"S0", "S1", "S2", "S1_obs", "S2_obs", "S2_act", "RP", "RA"}


def _outage_response(prev, nxt, alt_names):
    n0, a0, _ = prev
    n1, a1, _ = nxt
    return n1 == ESCALATE or (n1 == n0 and a1 == a0) or (n1 in alt_names and a1 == a0)


def false_outage_events(r):
    ev = []
    calls = r["calls"]
    alts = {calls[i][0] for i in range(len(calls)) if calls[i][0].endswith(("_backup", "_replica"))}
    for i in range(len(calls) - 1):
        if obs_kind(calls[i][2]) == "native" and calls[i][0] != ESCALATE and i + 1 >= r.get("n_forced", 0):
            ev.append(_outage_response(calls[i], calls[i + 1], alts | {calls[i][0] + "_backup", calls[i][0] + "_replica"}))
    if calls and obs_kind(calls[-1][2]) == "native":
        ev.append(False)                       # final real error with no response: not an outage response
    return ev


def summarize(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["suite"]].append(r)
    out = {"n": len(rows), "success": {k: float(np.mean([x["correct"] for x in v])) for k, v in sorted(by.items())},
           "n_by_suite": {k: len(v) for k, v in sorted(by.items())}}
    out["fault_free"] = out["success"].get("S0")
    s0 = {r["task_id"]: r["correct"] for r in by.get("S0", [])}
    cr = [r["correct"] for r in rows if r["suite"] in ("S1", "S2") and s0.get(r["task_id"])]
    out["conditional_recovery"] = float(np.mean(cr)) if cr else None
    gr = [r["correct"] for r in rows if r["suite"] in ("RP", "RA")]
    out["genuine_repair"] = float(np.mean(gr)) if gr else None
    ir = [r["correct"] for r in rows if r["suite"] in ("S1", "S2")]
    out["injected_recovery"] = float(np.mean(ir)) if ir else None
    out["correct_escalation"] = out["success"].get("S3")
    pa = [r["escalated"] for r in rows if r["suite"] in SOLVABLE]
    out["premature_abstention"] = float(np.mean(pa)) if pa else None
    fo = [e for r in rows for e in false_outage_events(r)]
    out["false_outage_rate"] = float(np.mean(fo)) if fo else None
    out["n_real_error_events"] = len(fo)
    out["overall"] = float(np.mean(list(out["success"].values()))) if out["success"] else None
    out["model_calls"] = int(sum(r.get("n_model_calls", 0) for r in rows))
    out["tokens_in"] = int(sum(r.get("tok_in", 0) for r in rows)); out["tokens_out"] = int(sum(r.get("tok_out", 0) for r in rows))
    return out


def metric_fn(name):
    def f(rows):
        s = summarize(rows)
        return s[name] if name in s else s["success"].get(name)
    return f


def cluster_boot(rows, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = collections.defaultdict(list)
    for r in rows:
        g[r["base_id"]].append(r)
    keys = list(g)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        v = fn([x for i in pick for x in g[keys[i]]])
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def paired_diff(rows_a, rows_b, fn, B=2000, seed=0):
    """Paired cluster bootstrap of fn(A) - fn(B) over the SAME episode specs (both policies on identical episodes)."""
    key = lambda r: (r["task_id"], r["suite"], r["seed"], r.get("state"))  # noqa: E731
    ka, kb = {key(r): r for r in rows_a}, {key(r): r for r in rows_b}
    common = sorted(set(ka) & set(kb))
    A = [ka[k] for k in common]; Bv = [kb[k] for k in common]
    g = collections.defaultdict(list)
    for i, r in enumerate(A):
        g[r["base_id"]].append(i)
    keys = list(g)
    rng = np.random.default_rng(seed)
    point = fn(A) - fn(Bv)
    vals = []
    for _ in range(B):
        idx = [i for j in rng.integers(0, len(keys), len(keys)) for i in g[keys[j]]]
        a, b = fn([A[i] for i in idx]), fn([Bv[i] for i in idx])
        if a is not None and b is not None:
            vals.append(a - b)
    return point, (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def retention(rows, ref_rows):
    ref = {r["task_id"]: r["correct"] for r in ref_rows if r["suite"] == "S0"}
    cur = {r["task_id"]: r["correct"] for r in rows if r["suite"] == "S0"}
    keep = [cur[t] for t, ok in ref.items() if ok and t in cur]
    return float(np.mean(keep)) if keep else None
