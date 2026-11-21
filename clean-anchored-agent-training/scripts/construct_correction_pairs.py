"""Construct outcome-verified corrective pairs, twin anchors and generic anchors (dataset cacrt_pairs_v1).

CPU stage (this script, default): negatives are the outage responses (retry / alternative / escalate) and outcomes are
verified under scripted continuations (`surrogate` = outage-rule policy, `repairer` = competent oracle).
GPU stage (--policy-samples FILE): negatives are the policy's own sampled actions at each state, read from FILE
(produced by `scripts/evaluate.py --mode sample_states`), verified with policy continuations via verify_stochastic.

Only `train` tasks produce training data. dev/test/heldout states are written as EVALUATION episodes (RP/RA suites)
and never enter training files.
Outputs: data/cacrt_pairs_v1/{train_states,train_anchors_twin,train_anchors_generic,eval_states_<split>}.jsonl
         data/cacrt_pairs_v1/MANIFEST.json (counts, env executions, sha256 of every file)
"""
import argparse
import collections
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.counterfactuals.construct import (Counter, generic_anchors, ra_states, rp_states, state_key,  # noqa: E402
                                           twin_anchors, verify_state)

VERSION = "cacrt_pairs_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(ROOT, "data", "tasks_v1.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", VERSION))
    ap.add_argument("--generic-per-twin", type=float, default=1.0, help="generic anchors per twin anchor (budget match)")
    args = ap.parse_args()
    tasks = [t for t in json.load(open(args.tasks)) if t["usable"]]
    os.makedirs(args.out, exist_ok=True)
    man = {"version": VERSION, "tasks_file_sha256": hashlib.sha256(open(args.tasks, "rb").read()).hexdigest(),
           "splits": {}, "env_executions": {}}
    for split in ("train", "dev", "test", "heldout"):
        T = [t for t in tasks if t["split"] == split]
        c = Counter()
        states, twins = [], []
        for t in T:
            for s in rp_states(t, c) + ra_states(t, c):
                s = verify_state(t, s, c)
                s["split"], s["base_id"] = split, t["base_id"]
                s["key"] = state_key(s)
                states.append(s)
                if split in ("train", "dev") and s["retained_pairs"]:
                    twins += twin_anchors(t, s, c)
        seen, uniq = set(), []
        for s in states:                       # identical (task, forced prefix) states are kept once
            k = (s["task_id"], json.dumps(s["forced"]))
            if k not in seen:
                seen.add(k); uniq.append(s)
        states = uniq
        st = collections.Counter()
        for s in states:
            st[f"{s['kind']}:total"] += 1
            st[f"{s['kind']}:naive_fails"] += s["naive_fails"]
            st[f"{s['kind']}:any_retained_pair"] += bool(s["retained_pairs"])
            for nm in s["retained_pairs"]:
                st[f"{s['kind']}:pair_{nm}"] += 1
            sur, rep = s["outcomes"]["surrogate"], s["outcomes"]["repairer"]
            st[f"{s['kind']}:pos_ok_surrogate"] += sur["pos"]
            for nm in ("retry", "escalate", "alternative"):
                if nm in rep:
                    st[f"{s['kind']}:neg_{nm}_fails_under_repairer"] += (not rep[nm])
        info = {"tasks": len(T), "base_tasks": len({t["base_id"] for t in T}), "states": dict(sorted(st.items())),
                "base_tasks_with_state": len({s["base_id"] for s in states if s["naive_fails"]})}
        name = "train_states" if split == "train" else f"eval_states_{split}"
        with open(os.path.join(args.out, name + ".jsonl"), "w", newline="\n", encoding="utf-8") as f:
            for s in states:
                f.write(json.dumps(s, default=str) + "\n")
        if split == "dev":                      # dev twins: model selection / retention curves only, never trained on
            with open(os.path.join(args.out, "dev_anchors_twin.jsonl"), "w", newline="\n", encoding="utf-8") as f:
                for r in twins:
                    f.write(json.dumps(r, default=str) + "\n")
            info["twin_anchors"] = dict(collections.Counter(a["kind"] for a in twins))
        if split == "train":
            ids_with_twin = {a["task_id"] for a in twins}
            gen = generic_anchors(T, int(round(len(twins) * args.generic_per_twin)), c, seed=0, exclude_ids=())
            for nm, rows in (("train_anchors_twin", twins), ("train_anchors_generic", gen)):
                with open(os.path.join(args.out, nm + ".jsonl"), "w", newline="\n", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, default=str) + "\n")
            info["twin_anchors"] = dict(collections.Counter(a["kind"] for a in twins))
            info["generic_anchors"] = len(gen)
            info["tasks_with_twin"] = len(ids_with_twin)
        man["splits"][split] = info
        man["env_executions"][split] = c.n
        print(split, json.dumps(info))
    man["files"] = {fn: hashlib.sha256(open(os.path.join(args.out, fn), "rb").read()).hexdigest()
                    for fn in sorted(os.listdir(args.out)) if fn.endswith(".jsonl")}
    json.dump(man, open(os.path.join(args.out, "MANIFEST.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
