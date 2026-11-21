"""Render the two matched SFT arms of the evidence-grounded study (docs/method_egr.md). GPU host; real tokenizer.

  AAC2: chosen action of every unique (a)/(b) state, PLAIN (no rationale)        + S1 chosen + 84 B-anchor chosen
  EGR : the same states and actions, each preceded by its one-sentence evidence  + identical S1 + B anchors
Identical states, actions, anchors and record counts; the only difference is the rationale text.
Usage: python scripts/render_egr_arms.py --tokenizer DIR --b-dpo <replication dpo.jsonl> --out DIR
"""
import argparse
import hashlib
import json
import os
import random

import _common
from verification.pairs import check_split_tokenization, load_jsonl, render_pair_texts

N_B_ANCHORS = 84


def uniq_states(P):
    seen, out = set(), []
    for p in P:
        if p["type"] in ("otv_a", "otv_b", "otv_s1") and p["state"] not in seen:
            seen.add(p["state"]); out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--b-dpo", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=os.path.join(_common.STUDY, "data", "otv_v2")); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D = a.data_dir
    from transformers import AutoTokenizer
    from src.training.run import load_context
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    _, _, tools_fn = load_context()
    plain, evid = uniq_states(load_jsonl(os.path.join(D, "pairs_plain.jsonl"))), uniq_states(load_jsonl(os.path.join(D, "pairs_evidence.jsonl")))
    assert [p["state"] for p in plain] == [p["state"] for p in evid], "arms must cover identical states"
    for x, y in zip(plain, evid):
        assert x["msgs"] == y["msgs"]
        cx, cy = dict(x["chosen"]), dict(y["chosen"])
        cy.pop("content", None)
        if "final_text" in cy:
            assert cy["final_text"].endswith(cx["final_text"])
        else:
            assert cx == cy, "same action, rationale only"
    banch = random.Random(0).sample(load_jsonl(a.b_dpo), N_B_ANCHORS)
    arms = {}
    for name, P in ((f"AAC2{a.tag}", plain), (f"EGR{a.tag}", evid)):
        recs = []
        for p in P:
            tools = tools_fn(p["task_id"])
            assert check_split_tokenization(tok, p["msgs"], tools, p["chosen"]), p["state"]
            pr, co = render_pair_texts(tok, p["msgs"], tools, p["chosen"])
            recs.append({"prompt": pr, "completion": co, "state": p["state"]})
        recs += [{"prompt": r["prompt"], "completion": r["chosen"], "state": r["state"]} for r in banch]
        arms[name] = recs
    os.makedirs(a.out, exist_ok=True)
    man = {"n_per_arm": len(arms[f"AAC2{a.tag}"]), "n_states": len(plain), "n_b_anchors": N_B_ANCHORS, "data_dir": D, "files": {}}
    for k, v in arms.items():
        p = os.path.join(a.out, f"{k}.jsonl")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man["files"][f"{k}.jsonl"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    man["completion_tokens_mean"] = {k: round(sum(len(tok(r["completion"], add_special_tokens=False)["input_ids"]) for r in v) / len(v), 1)
                                     for k, v in arms.items()}
    json.dump(man, open(os.path.join(a.out, f"MANIFEST{a.tag}.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
