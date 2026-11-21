"""Render the three matched method-study arms for TRL (GPU host; needs the real tokenizer).

All arms start from pi_B (= pi0 + B_s0 merged) and share the same anchors; they differ only in the AC data:
  OTV  (DPO): (a) error-state pairs + (b) twin post-check pairs + S1 anchors + 84 B pairs
  BAC  (DPO): (a) error-state pairs, cycled to |a|+|b| + S1 anchors + 84 B pairs      [CSO-style: critical step only]
  AAC  (SFT): chosen action of every unique (a)/(b) state, cycled to |a|+|b| + S1 chosen + 84 B chosen
Each arm has exactly the same number of records (hence identical steps). Every record passes the split-tokenization
check or the build aborts.
Usage: python scripts/render_otv_arms.py --tokenizer DIR --b-dpo <replication data/cts_rep/dpo.jsonl> --out DIR
"""
import argparse
import hashlib
import itertools
import json
import os
import random

import _common
from verification.pairs import check_split_tokenization, dpo_records, load_jsonl, render_pair_texts

N_B_ANCHORS = 84


def cycle_to(xs, n):
    return list(itertools.islice(itertools.cycle(xs), n))


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pairs", default=os.path.join(_common.STUDY, "data", "otv_v1", "pairs_abstract.jsonl"))
    ap.add_argument("--b-dpo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from src.training.run import load_context
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    _, _, tools_fn = load_context()
    P = load_jsonl(a.pairs)
    for p in P:
        for act in (p["chosen"], p["rejected"]):
            assert check_split_tokenization(tok, p["msgs"], tools_fn(p["task_id"]), act), p["state"]
    A = [p for p in P if p["type"] == "otv_a"]; Bp = [p for p in P if p["type"] == "otv_b"]; S1 = [p for p in P if p["type"] == "otv_s1"]
    n_ac = len(A) + len(Bp)
    bdpo = load_jsonl(a.b_dpo)
    banch = random.Random(0).sample(bdpo, N_B_ANCHORS)
    rec = lambda ps: dpo_records(tok, ps, tools_fn)
    arms = {"OTV": rec(A + Bp + S1) + banch,
            "BAC": rec(cycle_to(A, n_ac) + S1) + banch}
    # SFT: one record per unique state (the 3 (a) pairs of a state share the chosen action)
    seen, uniq = set(), []
    for p in A + Bp:
        if p["state"] not in seen:
            seen.add(p["state"]); uniq.append(p)
    sft = []
    for p in cycle_to(uniq, n_ac) + S1:
        pr, co = render_pair_texts(tok, p["msgs"], tools_fn(p["task_id"]), p["chosen"])
        sft.append({"prompt": pr, "completion": co, "state": p["state"]})
    sft += [{"prompt": r["prompt"], "completion": r["chosen"], "state": r["state"]} for r in banch]
    arms["AAC"] = sft
    assert len({len(v) for v in arms.values()}) == 1, {k: len(v) for k, v in arms.items()}
    os.makedirs(a.out, exist_ok=True)
    man = {"n_per_arm": len(sft), "n_a": len(A), "n_b": len(Bp), "n_s1": len(S1), "n_b_anchors": N_B_ANCHORS,
           "n_unique_ac_states": len(uniq), "source_pairs_sha256": hashlib.sha256(open(a.pairs, "rb").read()).hexdigest(),
           "b_dpo_sha256": hashlib.sha256(open(a.b_dpo, "rb").read()).hexdigest(), "files": {}}
    for k, v in arms.items():
        man["files"][f"{k}.jsonl"] = write(os.path.join(a.out, f"{k}.jsonl"), v)
    man["prompt_tokens_mean"] = {k: round(sum(len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) for r in v[:50]) / 50) for k, v in arms.items()}
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
