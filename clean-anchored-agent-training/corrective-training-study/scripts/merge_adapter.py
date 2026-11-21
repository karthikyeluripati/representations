"""Merge a LoRA adapter into its base and save a standalone model (used to build pi_B = pi0 + B_s0).
Usage: python scripts/merge_adapter.py --base /workspace/models/pi0 --adapter <dir> --out /workspace/models/piB"""
import argparse
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
tmp = a.out + ".tmp"
shutil.rmtree(tmp, ignore_errors=True)
m.save_pretrained(tmp, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(tmp)
for f in ("chat_template.jinja", "generation_config.json"):
    if os.path.exists(os.path.join(a.base, f)) and not os.path.exists(os.path.join(tmp, f)):
        shutil.copy(os.path.join(a.base, f), tmp)
os.replace(tmp, a.out)
print("merged ->", a.out)
