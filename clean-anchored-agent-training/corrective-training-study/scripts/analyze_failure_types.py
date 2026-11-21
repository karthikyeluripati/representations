"""Behavioural failure taxonomy for saved evaluation traces (executed calls + observations + grader outcome).

Every FAILED episode gets exactly one primary label, by this precedence (first match wins); all matching labels are
also recorded so overlap can be audited:
  H interface   a call to an unknown function, a malformed call, or a Python signature error (TypeError/unexpected kwarg)
  B false_outage a REAL error is followed by an outage response (identical retry, alternative endpoint, or escalate)
  E premature_escalation  escalated in a solvable episode without a preceding real error (S0/S1/S2/shift suites/RP/RA)
  X missed_escalation     S3 (impossible) episode that did not escalate
  D ordering    a real precondition-type error occurred on an agent call and the final state is wrong (not B)
  C argument    final state wrong and some agent call has a ground-truth function name with different arguments
  F unneeded_retry  an identical state-changing call repeated after it succeeded
  G general     anything else (incomplete, wrong function, stopped early, step cap)
Successful episodes that met a real or injected error are counted as A (correct recovery).
Error TYPES are normalised real-error messages (digits/quoted values stripped); a type is "unseen" if it never occurs
in any TRAIN corrective state or train reference run.
Usage: python analyze_failure_types.py --traces <eval.jsonl or archive.tgz::member> [--name pi0]
"""
import argparse
import collections
import json
import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))
sys.path.insert(0, PROJ)

from src.environment.bfcl_env import ESCALATE, obs_kind  # noqa: E402
from src.replay.replay import parse_call  # noqa: E402

SOLVABLE = {"S0", "S1", "S2", "S1_obs", "S2_obs", "S2_act", "RP", "RA"}
PRECOND = re.compile(r"not (logged in|authenticated|running|started|locked|pressed)|must be|before|first|requires|no user|not found in current", re.I)


def load(spec):
    if "::" in spec:
        arch, member = spec.split("::")
        return [json.loads(l) for l in tarfile.open(arch).extractfile(member)]
    return [json.loads(l) for l in open(spec, encoding="utf-8")]


def err_type(o):
    try:
        d = json.loads(o)
        o = d.get("error", o) if isinstance(d, dict) else o
    except Exception:  # noqa: BLE001
        pass
    s = re.sub(r"'[^']*'|\"[^\"]*\"|\d+(\.\d+)?", "_", str(o))
    return re.sub(r"\s+", " ", s).strip()[:90]


def is_outage_resp(prev, nxt):
    return nxt[0] == ESCALATE or (nxt[0] == prev[0] and nxt[1] == prev[1]) or (nxt[0] in (prev[0] + "_backup", prev[0] + "_replica") and nxt[1] == prev[1])


def labels(r, task):
    calls = [tuple(c) for c in r["calls"]]
    nf = r.get("n_forced", 0)
    own = calls[nf:]
    L = set()
    if any(o.startswith("Error: unknown function") or "malformed" in o or "TypeError" in o or "unexpected keyword" in o for _, _, o in own):
        L.add("H")
    for i in range(max(nf - 1, 0), len(calls) - 1):
        if obs_kind(calls[i][2]) == "native" and calls[i][0] != ESCALATE and is_outage_resp(calls[i], calls[i + 1]):
            L.add("B")
    native_before_esc = any(obs_kind(o) == "native" for _, _, o in calls)
    if r["suite"] in SOLVABLE and r["escalated"] and not native_before_esc:
        L.add("E")
    if r["suite"] == "S3" and not r["escalated"]:
        L.add("X")
    if any(obs_kind(o) == "native" and PRECOND.search(o) for _, _, o in own) and not r["state_ok"]:
        L.add("D")
    gt = {}
    for c in task["turn_gt"]:
        n, kw, pos = parse_call(c)
        gt.setdefault(n, []).append((kw, pos))
    if not r["state_ok"] and any(n in gt and all(a != kw for kw, _ in gt[n]) for n, a, _ in own if n != ESCALATE):
        L.add("C")
    seen = set()
    for n, a, o in own:
        k = (n, json.dumps(a, sort_keys=True))
        if k in seen and obs_kind(o) == "ok" and not n.startswith(("get_", "display", "list", "check")):
            L.add("F")
        if obs_kind(o) == "ok":
            seen.add(k)
    return L


ORDER = ["H", "B", "E", "X", "D", "C", "F"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--tasks", default=os.path.join(PROJ, "data", "tasks_v1.json"))
    ap.add_argument("--train-states", default=os.path.join(PROJ, "data", "cacrt_pairs_v1", "train_states.jsonl"))
    ap.add_argument("--name", default="policy")
    ap.add_argument("--out", default=os.path.join(STUDY, "results", "failure_types"))
    a = ap.parse_args()
    tasks = {t["id"]: t for t in json.load(open(a.tasks, encoding="utf-8"))}
    rows = load(a.traces)
    train_types = {err_type(json.loads(l)["error_obs"]) for l in open(a.train_states, encoding="utf-8")}
    res = collections.defaultdict(lambda: collections.Counter())
    over = collections.defaultdict(collections.Counter)
    etypes = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        sp = r["split"]
        L = labels(r, tasks[r["task_id"]])
        if r["correct"]:
            met = any(obs_kind(o) in ("native", "injected") for _, _, o in r["calls"])
            res[sp]["A_correct_recovery" if met else "ok_no_error"] += 1
        else:
            prim = next((x for x in ORDER if x in L), "G")
            res[sp][prim] += 1
            over[sp][len(L)] += 1
        # real-error events by type: did the agent's next action treat it as an outage?
        calls = r["calls"]
        for i in range(len(calls) - 1):
            if obs_kind(calls[i][2]) == "native" and calls[i][0] != ESCALATE and i + 1 >= r.get("n_forced", 0):
                ty = err_type(calls[i][2])
                k = "seen" if ty in train_types else "unseen"
                etypes[sp][(k, "outage_resp" if is_outage_resp(tuple(calls[i]), tuple(calls[i + 1])) else "other")] += 1
    out = {"name": a.name, "traces": a.traces,
           "labels": {sp: dict(c) for sp, c in res.items()},
           "n_labels_per_failure": {sp: dict(c) for sp, c in over.items()},
           "real_error_events_by_type_novelty": {sp: {f"{k[0]}:{k[1]}": v for k, v in c.items()} for sp, c in etypes.items()}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"{a.name}.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
