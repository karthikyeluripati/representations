"""Context-free ("action-only") embeddings of candidate-action strings, for the Phase 3 ablation.

Each action string is encoded ALONE (no system prompt, no observation, no history), and the last-layer
hidden states are mean-pooled over its tokens. This is the same pooling as the stored contextual action
embeddings, but without the state, so a probe on these features can only learn an action-text prior.

Usage: python scripts/extract_action_text_emb.py --model Qwen/Qwen2.5-3B-Instruct \
           --candidates cands_3b.json --out results/phase3/action_text_emb_3b.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", required=True, help="JSON list of action strings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    cands = sorted(set(json.load(open(a.candidates))))
    tok = AutoTokenizer.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    embs = []
    for i in range(0, len(cands), a.batch):
        b = cands[i:i + a.batch]
        enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            h = m(**enc, output_hidden_states=True).hidden_states[-1].float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        embs.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    E = np.concatenate(embs)
    np.savez(a.out, candidates=np.array(cands), emb=E.astype(np.float32), model=np.array(a.model))
    print(f"wrote {E.shape} for {len(cands)} unique actions -> {a.out}")


if __name__ == "__main__":
    main()
