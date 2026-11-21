"""Idempotency-aware recovery (IAR) training decisions, fold T (docs/method_iar.md). Tokenizer-free, execution-verified.

Train classes MessageAPI, TradingBot, TwitterAPI, TicketAPI; train split only. Every record = (state msgs, chosen action):
  ac_twin   decisive write tasks, wordings explicit/train1/train2 (as AAC2) AND the timeout-type TRAIN outage strings
            t_gateway/t_dropped: check at the error; after the check, continue (ACa) or re-issue (ACl)
  rejection decisive write tasks, the "503" TRAIN string (a clean rejection, nothing applied): retry directly
  idem      idempotent fault targets (re-execution leaves the state unchanged), any TRAIN outage string: retry directly
Baseline (old convention, count-matched): ac_twin + write_retry (decisive writes, ALL three TRAIN strings, retry
directly, as the S1 anchors of AAC2) + the same idem records, cycled to IAR's size. The two sets differ ONLY in how a
timeout-type failure of a state-changing call is answered (IAR: check; baseline: retry).
Usage: python scripts/build_iar_data.py
"""
import collections
import hashlib
import itertools
import json
import os

import _common
from build_otv_pairs import verifier_for
from src.counterfactuals.construct import SYSTEM, episode_messages
from src.environment.bfcl_env import AC_TRAIN_MSGS, IAR_AMBIG_TRAIN_MSGS, TRAIN_ERRORS, Episode, load_bfcl
from verification.outcome_twins import _bound, _do, build_task_pairs, fault_index, rest_actions, run_plan

CLASSES = ("MessageAPI", "TradingBot", "TwitterAPI", "TicketAPI")
P, S = _common.PROJ, _common.STUDY
OUT = os.path.join(S, "data", "iar_v1")


def retry_state(task, err):
    ep = Episode(task, scenario="S1", seed=0, error_pool=[err], k=1)
    for c in task["turn_gt"][:fault_index(task)]:
        _do(ep, _bound(ep, c))
    f = _bound(ep, task["turn_gt"][fault_index(task)])
    obs = _do(ep, f)
    assert obs == err
    if not run_plan(ep, lambda e: [f] + rest_actions(e, task)):
        return None
    return {"task_id": task["id"], "msgs": episode_messages(task, ep, SYSTEM), "chosen": f}


def uniq_chosen(pairs, kind):
    seen, out = set(), []
    for p in pairs:
        if p["type"] in ("otv_a", "otv_b") and p["state"] not in seen:
            seen.add(p["state"])
            out.append({"kind": kind, "task_id": p["task_id"], "msgs": p["msgs"], "chosen": p["chosen"], "state": p["state"]})
    return out


def main():
    tasks = {t["id"]: t for t in json.load(open(os.path.join(P, "data", "tasks_v1.json"), encoding="utf-8"))}
    feas = {r["task_id"]: r for r in json.load(open(os.path.join(P, "results", "feasibility_ambiguous_commit.json"), encoding="utf-8"))}
    docs = load_bfcl()[2]
    train = [t for t in tasks.values() if t["usable"] and t["split"] == "train" and t["fault_class"] in CLASSES]
    dec = [t for t in train if feas[t["id"]]["decisive"]]
    idem = [t for t in train if not feas[t["id"]]["non_idempotent"]]
    base, ambig, rej, idm, wret, drops = [], [], [], [], [], collections.Counter()
    for t in dec:
        ver = verifier_for(t, feas[t["id"]], docs)
        for m in AC_TRAIN_MSGS:
            ps, rep = build_task_pairs(t, ver, m); drops.update(rep["dropped"])
            base += uniq_chosen(ps, "ac_twin")
        for m in IAR_AMBIG_TRAIN_MSGS:
            ps, rep = build_task_pairs(t, ver, m); drops.update(rep["dropped"])
            ambig += uniq_chosen(ps, "ac_twin_timeout")
        for j, e in enumerate(TRAIN_ERRORS):
            r = retry_state(t, e)
            if not r:
                drops["write_retry:unverified"] += 1; continue
            wret.append({"kind": "write_retry", **r, "state": f"{t['id']}|wret|{j}"})
            if j == 0:
                rej.append({"kind": "rejection_retry", **r, "state": f"{t['id']}|rej"})
    for t in idem:
        for j, e in enumerate(TRAIN_ERRORS):
            r = retry_state(t, e)
            if r:
                idm.append({"kind": "idem_retry", **r, "state": f"{t['id']}|idem|{j}"})
            else:
                drops["idem:unverified"] += 1
    iar = base + ambig + rej + idm
    old = base + wret + idm
    aac = list(itertools.islice(itertools.cycle(old), len(iar)))         # count-matched baseline (old convention)
    os.makedirs(OUT, exist_ok=True)
    man = {"version": "iar_v1", "classes": CLASSES, "n_decisive_tasks": len(dec), "n_idempotent_tasks": len(idem),
           "counts": dict(collections.Counter(r["kind"] for r in iar)), "n": len(iar), "dropped": dict(drops),
           "baseline_counts_before_cycling": dict(collections.Counter(r["kind"] for r in old))}
    for name, rows in (("iar.jsonl", iar), ("aac_matched.jsonl", aac)):
        p = os.path.join(OUT, name)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
