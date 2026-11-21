"""Baseline A: balanced corrective SFT (TRL SFTTrainer, completion-only loss), matched to Baseline B in examples, epochs and batch.

Usage: python scripts/train_balanced_sft.py --model /workspace/models/pi0 --lr <dev-selected> --seed 0 --out results/runs/<name>
Defaults follow CSO's public LlamaFactory command (lr 5e-6, batch 4, 3 epochs) and LlamaFactory LoRA defaults (r 8).
Resumes automatically from the newest checkpoint in --out.
"""
import argparse
import hashlib
import json
import os
import time

import _common  # noqa: F401
from training.trl_runner import train_sft, load_model
from verification.pairs import load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default=os.path.join(_common.STUDY, "data", "cts_v1", "sft_balanced.jsonl"))
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lora-r", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-steps-bench", type=int, default=0, help="throughput benchmark: stop after N steps")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
recs = load_jsonl(a.data)
json.dump({**vars(a), "data_sha256": hashlib.sha256(open(a.data, "rb").read()).hexdigest(), "n": len(recs)},
          open(os.path.join(a.out, "run_config.json"), "w"), indent=1)
model, tok = load_model(a.model)
t0 = time.time()
if a.max_steps_bench:
    recs = recs[: a.max_steps_bench * a.batch]
    a.epochs = 1
train_sft(model, tok, recs, a.out, a.lr, a.seed, epochs=a.epochs, batch=a.batch, r=a.lora_r)
json.dump({"train_seconds": time.time() - t0, "n_records": len(recs)}, open(os.path.join(a.out, "timing.json"), "w"))
