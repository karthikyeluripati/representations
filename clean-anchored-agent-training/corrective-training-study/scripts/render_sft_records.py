"""Render abstract SFT decisions ({task_id, msgs, chosen}) to TRL prompt/completion records + the shared B anchors.
Usage: python scripts/render_sft_records.py --tokenizer DIR --b-dpo FILE --out-dir DIR IN=NAME [IN=NAME ...]"""
import argparse
import hashlib
import json
import os
import random

import _common  # noqa: F401
from verification.pairs import check_split_tokenization, load_jsonl, render_pair_texts

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True); ap.add_argument("--b-dpo", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--n-anchors", type=int, default=84); ap.add_argument("sets", nargs="+")
a = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402
from src.training.run import load_context  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.tokenizer)
_, _, tools_fn = load_context()
banch = random.Random(0).sample(load_jsonl(a.b_dpo), a.n_anchors)
os.makedirs(a.out_dir, exist_ok=True)
man = {}
for spec in a.sets:
    src, name = spec.split("=")
    recs = []
    for r in load_jsonl(src):
        tools = r.get("tools") or tools_fn(r["task_id"])          # synthetic records carry their own tool schemas
        assert check_split_tokenization(tok, r["msgs"], tools, r["chosen"]), r["state"]
        pr, co = render_pair_texts(tok, r["msgs"], tools, r["chosen"])
        recs.append({"prompt": pr, "completion": co, "state": r["state"]})
    recs += [{"prompt": b["prompt"], "completion": b["chosen"], "state": b["state"]} for b in banch]
    p = os.path.join(a.out_dir, name + ".jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man[name] = {"n": len(recs), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "source": src}
json.dump(man, open(os.path.join(a.out_dir, "MANIFEST_render.json"), "w"), indent=1)
print(json.dumps(man, indent=1))
