"""Fingerprint a machine's LM numerics on fixed, deterministic ALFWorld decision contexts.

Run on two machines with the same config and compare: if every score/embedding is bitwise
equal, data produced on either machine comes from the *same* frozen policy (bf16 greedy
decisions are sensitive to numerics, so this must be checked, not assumed).

Usage:  python scripts/numerics_fingerprint.py --config configs/baseline_pod.yaml --out fp_A.npz
        python scripts/numerics_fingerprint.py --compare fp_A.npz fp_B.npz
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402


def make(cfg_path: str, out: str, n_games: int = 6, n_steps: int = 6):
    from dare.agents.demos import build_demos
    from dare.agents.lm import LMScorer, build_prompt_messages, static_user_prefix
    from dare.environments.alfworld_env import AlfWorldEnv, list_games, select_games
    from dare.utils.common import load_config

    cfg = load_config("configs/environment.yaml", cfg_path)
    m, pc = cfg["model"], cfg["policy"]
    s = LMScorer(m["id"], m.get("dtype", "float32"), m.get("num_threads"), m.get("max_batch", 16),
                 m.get("device", "auto"), m.get("reuse_prefix", True))
    demos = build_demos(cfg["env"]["data_root"], pc["demo_cache"]) if pc.get("few_shot") else {}
    env = AlfWorldEnv(max_steps=50)
    rec = {}
    for gi, g in enumerate(select_games(list_games(cfg["env"]["data_root"], "valid_seen"), n_games)):
        rng = random.Random(gi)
        st = env.reset(g.game_file)
        intro, hist = st.observation, []
        demo = demos.get(g.task_family)
        for t in range(n_steps):
            r = s.score(build_prompt_messages(intro, hist, st.observation, st.admissible, pc.get("max_history", 8),
                                              demo), st.admissible, static_user_prefix(intro, demo))
            rec[f"g{gi}_t{t}_logp"] = r.logp_sum
            rec[f"g{gi}_t{t}_emb"] = r.action_emb.astype(np.float32)
            a = rng.choice(st.admissible)
            st = env.step(a)
            hist.append((a, st.observation))
    import torch

    rec["_device"] = np.array(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    rec["_torch"] = np.array(torch.__version__)
    np.savez(out, **rec)
    print(f"wrote {len(rec) - 2} arrays to {out} on {rec['_device']}")


def compare(a_path: str, b_path: str):
    a, b = np.load(a_path), np.load(b_path)
    keys = sorted(k for k in a.files if not k.startswith("_"))
    assert keys == sorted(k for k in b.files if not k.startswith("_")), "different contexts"
    exact = sum(np.array_equal(a[k], b[k]) for k in keys)
    worst = max(float(np.abs(a[k] - b[k]).max()) for k in keys)
    argmax_same = sum(int(np.argmax(a[k]) == np.argmax(b[k])) for k in keys if k.endswith("_logp"))
    n_logp = sum(k.endswith("_logp") for k in keys)
    print(f"A: {a['_device']} torch {a['_torch']}  B: {b['_device']} torch {b['_torch']}")
    print(f"bitwise-equal arrays: {exact}/{len(keys)}; max |diff| {worst:.3g}; "
          f"greedy argmax agrees {argmax_same}/{n_logp}")
    print("IDENTICAL NUMERICS" if exact == len(keys) else "NUMERICS DIFFER")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    compare(*args.compare) if args.compare else make(args.config, args.out)
