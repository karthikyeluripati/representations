"""Build execution-verified outcome-twin pairs (abstract, tokenizer-free) for the OTV method study.

Split (pre-registered in docs/method_outcome_twins.md): TRAIN = DECISIVE diagnostic tasks of TravelAPI / TradingBot in the
train split; wordings = AC_TRAIN_MSGS only. Everything else (other classes, other splits, para/implicit) is evaluation.
Output: data/otv_v1/{pairs_abstract.jsonl, build_report.json, MANIFEST.json}
Usage: python scripts/build_otv_pairs.py
"""
import argparse
import collections
import hashlib
import json
import os

import _common
from src.environment.bfcl_env import AC_TRAIN_MSGS, load_bfcl
from src.replay.replay import bind_args, new_instances, parse_call, run_calls
from verification.outcome_twins import build_s1_anchor, build_task_pairs, fault_index

TRAIN_CLASSES = ("TravelAPI", "TradingBot")                         # v1; v2 passes --train-classes
P, S = _common.PROJ, _common.STUDY


def verifier_for(task, row, docs):
    i = fault_index(task)
    n, kw, pos = parse_call(task["turn_gt"][i])
    inst = new_instances(task["entry"]); run_calls(inst, task["prefix_gt"] + task["turn_gt"][:i])
    fa = bind_args(inst, n, kw, pos)
    d = next(d for d in docs[task["fault_class"]] if d["name"] == row["decisive_verifiers"][0])
    return d["name"], {q: fa[q] for q in d.get("parameters", {}).get("required", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="otv_v1")
    ap.add_argument("--train-classes", default=",".join(TRAIN_CLASSES))
    ap.add_argument("--evidence", action="store_true", help="evidence-grounded rationales on (a) chosen and (b) both sides")
    ap.add_argument("--out-name", default="pairs_abstract.jsonl")
    a = ap.parse_args()
    classes = tuple(a.train_classes.split(","))
    OUT = os.path.join(S, "data", a.version)
    tasks = {t["id"]: t for t in json.load(open(os.path.join(P, "data", "tasks_v1.json"), encoding="utf-8"))}
    rows = json.load(open(os.path.join(P, "results", "feasibility_ambiguous_commit.json"), encoding="utf-8"))
    _, _, docs = load_bfcl()
    train = [r for r in rows if r["decisive"] and r["split"] == "train" and r["fault_class"] in classes]
    pairs, report = [], {"n_train_tasks": len(train), "dropped": collections.Counter(), "per_task": {}}
    for r in train:
        t = tasks[r["task_id"]]
        ver = verifier_for(t, r, docs)
        for m in AC_TRAIN_MSGS:
            ps, rep = build_task_pairs(t, ver, m, evidence=a.evidence)
            pairs += ps
            report["dropped"].update(rep["dropped"])
            report["per_task"][f"{t['id']}|{m}"] = {"n": len(ps), "outcomes": rep["outcomes"]}
        anc = build_s1_anchor(t)
        if anc:
            pairs.append(anc)
        else:
            report["dropped"]["s1:unverified"] += 1
    c = collections.Counter(p["type"] + ":" + p["neg"] for p in pairs)
    report["counts"] = dict(c); report["dropped"] = dict(report["dropped"]); report["n_pairs"] = len(pairs)
    report["train_task_ids"] = [r["task_id"] for r in train]
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, a.out_name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for x in pairs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if a.version == "otv_v1":            # v1 layout kept byte-compatible with the seed-0 run
        json.dump(report, open(os.path.join(OUT, "build_report.json"), "w", encoding="utf-8", newline="\n"), indent=1)
        json.dump({"version": "otv_v1", "train_classes": classes, "train_msgs": AC_TRAIN_MSGS, "n_pairs": len(pairs),
                   "counts": dict(c), "pairs_abstract.jsonl": sha},
                  open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    else:
        stem = a.out_name.replace(".jsonl", "")
        json.dump(report, open(os.path.join(OUT, f"build_report_{stem}.json"), "w", encoding="utf-8", newline="\n"), indent=1)
        mp = os.path.join(OUT, "MANIFEST.json")
        man = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
        man.update({"version": a.version, "train_classes": classes, "train_msgs": AC_TRAIN_MSGS,
                    a.out_name: {"n_pairs": len(pairs), "counts": dict(c), "evidence": a.evidence, "sha256": sha}})
        json.dump(man, open(mp, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: report[k] for k in ("n_train_tasks", "n_pairs", "counts", "dropped")}, indent=1))


if __name__ == "__main__":
    main()
