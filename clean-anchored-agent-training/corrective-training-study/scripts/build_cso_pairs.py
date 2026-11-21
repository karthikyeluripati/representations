"""Build the two matched baseline datasets (version cts_v1) from TRAIN data only.

  data/cts_v1/dpo.jsonl          Baseline B: the 567 pi0-verified corrective pairs (252 states), rendered for TRL
  data/cts_v1/sft_balanced.jsonl Baseline A: 567 verified-correct decisions = 189 repair + 189 outage + 189 fault-free
  data/cts_v1/MANIFEST.json      counts, token counts (real tokenizer), split-tokenization check results, sha256s
Every record passes `check_split_tokenization` (TRL's separate prompt/completion tokenization == the model's real
input under the chat template) or the build aborts.
Usage: python scripts/build_cso_pairs.py --tokenizer <Qwen3-4B-Instruct-2507 dir> [--sample N  (CPU smoke)]
"""
import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(STUDY, "src"))

from verification.pairs import (balanced_sft_decisions, check_split_tokenization, dpo_records,  # noqa: E402
                                    load_jsonl, render_pair_texts)

VERSION = "cts_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pairs", default=os.path.join(STUDY, "data", "policy_pairs_v1.jsonl"))
    ap.add_argument("--out", default=os.path.join(STUDY, "data", VERSION))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="full build and checks, print stats, write nothing")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from src.training.run import load_context
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tasks, by_id, tools_fn = load_context()
    train = [t for t in tasks if t["usable"] and t["split"] == "train"]
    pairs = load_jsonl(a.pairs)
    assert all(by_id[p["task_id"]]["split"] == "train" for p in pairs)   # 567 for the archived pi0; re-verified pairs may differ
    n_cat = len(pairs) // 3
    dec, avail = balanced_sft_decisions(train, pairs, n_cat, seed=a.seed)
    if a.sample:
        pairs, dec = pairs[:a.sample], dec[:a.sample]
    bad = 0
    for p in pairs:
        for act in (p["chosen"], p["rejected"]):
            bad += not check_split_tokenization(tok, p["msgs"], tools_fn(p["task_id"]), act)
    sft = []
    for d in dec:
        tools = tools_fn(d["task_id"])
        bad += not check_split_tokenization(tok, d["msgs"], tools, d["action"])
        pr, co = render_pair_texts(tok, d["msgs"], tools, d["action"])
        sft.append({"prompt": pr, "completion": co, "category": d["category"], "task_id": d["task_id"]})
    if bad:
        raise SystemExit(f"{bad} records fail the split-tokenization check")
    dpo = dpo_records(tok, pairs, tools_fn)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    stats = {"dpo": {"n": len(dpo), "states": len({x["state"] for x in dpo}), "by_negative": dict(collections.Counter(x["neg"].split("+")[0] if x["neg"].startswith("cpu_") else "sampled" for x in dpo)),
                     "prompt_tokens": sum(ntok(x["prompt"]) for x in dpo), "chosen_tokens": sum(ntok(x["chosen"]) + 1 for x in dpo),
                     "rejected_tokens": sum(ntok(x["rejected"]) + 1 for x in dpo), "max_prompt": max(ntok(x["prompt"]) for x in dpo)},
             "sft": {"n": len(sft), "by_category": dict(collections.Counter(x["category"] for x in sft)), "available_by_category": avail,
                     "prompt_tokens": sum(ntok(x["prompt"]) for x in sft), "completion_tokens": sum(ntok(x["completion"]) + 1 for x in sft),
                     "max_prompt": max(ntok(x["prompt"]) for x in sft), "distinct_tasks": len({x["task_id"] for x in sft})},
             "split_tokenization_failures": 0}
    if a.sample or a.dry_run:
        print(json.dumps(stats, indent=1)); return
    os.makedirs(a.out, exist_ok=True)
    files = {}
    for name, rows in (("dpo.jsonl", dpo), ("sft_balanced.jsonl", sft)):
        p = os.path.join(a.out, name)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    man = {"version": VERSION, "source_pairs_sha256": hashlib.sha256(open(a.pairs, "rb").read()).hexdigest(),
           "tasks_sha256": hashlib.sha256(open(os.path.join(PROJ, "data", "tasks_v1.json"), "rb").read()).hexdigest(),
           "seed": a.seed, "stats": stats, "files": files}
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
