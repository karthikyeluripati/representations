"""CAW-v2 training data (docs/method_caw2.md): CAW (data/caw_v1) + execution-verified PERSISTENT-FAILURE continuations.

CAW-v1 regressed on persistent outages (S2: use the backup endpoint) and impossible requests (S3: escalate) because
its data never showed what to do when the re-issued call fails again. Added on BFCL fold-T train tasks (decisive):
  persist_backup   S2-type: f -> timeout, check -> not applied, f -> timeout again   => chosen: f's backup endpoint
  persist_escalate S3-type: f -> timeout, check -> error, check -> error again       => chosen: escalate_to_user
Every chosen action is verified by executing its continuation under the BFCL grader. To keep the total at 1,349
(+84 anchors = 1,433, same as SYN64/CAW), the cycled BFCL part of CAW is shortened by the number of added records.
Output: data/caw2_v1/{caw2.jsonl, MANIFEST.json}
"""
import collections
import hashlib
import itertools
import json
import os

import _common
from build_otv_pairs import verifier_for
from src.counterfactuals.construct import SYSTEM, episode_messages
from src.environment.bfcl_env import ESCALATE, TRAIN_ERRORS, Episode, load_bfcl
from verification.outcome_twins import _bound, _do, fault_index, rest_actions, run_plan

P, S = _common.PROJ, _common.STUDY
OUT = os.path.join(S, "data", "caw2_v1")
CLASSES = ("MessageAPI", "TradingBot", "TwitterAPI", "TicketAPI")


def episode(t, scenario, err):
    i = fault_index(t)
    return Episode(t, scenario=scenario, seed=0, error_pool=[err], forced=list(t["turn_gt"][:i]), inject_from=i)


def main():
    tasks = {x["id"]: x for x in json.load(open(os.path.join(P, "data", "tasks_v1.json"), encoding="utf-8"))}
    feas = {r["task_id"]: r for r in json.load(open(os.path.join(P, "results", "feasibility_ambiguous_commit.json"), encoding="utf-8"))}
    docs = load_bfcl()[2]
    train = [tasks[r["task_id"]] for r in feas.values() if r["decisive"] and r["split"] == "train" and r["fault_class"] in CLASSES]
    add, drops = [], collections.Counter()
    for t in train:
        vn, va = verifier_for(t, feas[t["id"]], docs); ver = {"name": vn, "arguments": va}
        i = fault_index(t)
        for e in TRAIN_ERRORS[1:]:
            # S2-type: only f fails, persistently; the check works and shows "not applied"
            ep = episode(t, "S2", e); f = _bound(ep, t["turn_gt"][i])
            o1 = _do(ep, f); o2 = _do(ep, ver); o3 = _do(ep, f)
            bk = {"name": ep.alt, "arguments": f["arguments"]}
            if o1 == e and o3 == e and not str(o2).startswith("Error") and run_plan(ep, lambda x: [bk] + rest_actions(x, t)) \
                    and not run_plan(ep, lambda x: [f] + rest_actions(x, t)):
                add.append({"kind": "persist_backup", "task_id": t["id"], "msgs": episode_messages(t, ep, SYSTEM), "chosen": bk,
                            "state": f"{t['id']}|S2|{e[:12]}"})
            else:
                drops["persist_backup:unverified"] += 1
            # S3-type: every method of the class fails; the check fails twice
            ep = episode(t, "S3", e); f = _bound(ep, t["turn_gt"][i])
            o1 = _do(ep, f); o2 = _do(ep, ver); o3 = _do(ep, ver)
            esc = {"name": ESCALATE, "arguments": {"reason": "The service keeps failing, so the request cannot be completed right now."}}
            if o1 == e and o2 == e and o3 == e and run_plan(ep, lambda x: [esc]) and not run_plan(ep, lambda x: [f] + rest_actions(x, t)):
                add.append({"kind": "persist_escalate", "task_id": t["id"], "msgs": episode_messages(t, ep, SYSTEM), "chosen": esc,
                            "state": f"{t['id']}|S3|{e[:12]}"})
            else:
                drops["persist_escalate:unverified"] += 1
    caw = [json.loads(l) for l in open(os.path.join(S, "data", "caw_v1", "caw.jsonl"), encoding="utf-8")]
    syn = [r for r in caw if r["task_id"].startswith("synth:")]
    bfcl_unique = [json.loads(l) for l in open(os.path.join(S, "data", "iar_v1", "iar.jsonl"), encoding="utf-8")]
    bfcl_unique = [r for r in bfcl_unique if r["kind"] != "idem_retry"]
    bfcl = list(itertools.islice(itertools.cycle(bfcl_unique), 517 - len(add)))
    rows = bfcl + add + syn
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "caw2.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"version": "caw2_v1", "n": len(rows), "n_synthetic": len(syn), "n_persist": len(add), "n_bfcl_cycled": len(bfcl),
           "counts": dict(collections.Counter(r["kind"] for r in rows)), "dropped": dict(drops),
           "caw2.jsonl": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
