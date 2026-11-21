"""Measure every arm's data budget with the REAL Qwen3-4B-Instruct-2507 tokenizer and chat template (no weights needed).

Reports per arm: examples, examples dropped by --max-len, target tokens and total tokens per epoch, anchor tokens,
max sequence length, and the number of chat-template prefix-stability failures (must be 0: target masks are derived
from prefix rendering). Writes results/budget/arm_budgets.json.
Usage: python scripts/measure_budget.py --tokenizer <dir with tokenizer.json + tokenizer_config.json>
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.baselines.arms import ARMS  # noqa: E402
from src.training.render import HFRenderer  # noqa: E402
from src.training.run import build_records, build_val, load_context  # noqa: E402
from src.training.trainer import Encoder  # noqa: E402


def stats(enc, recs):
    n_ok, tgt, tot, mx, fail = 0, 0, 0, 0, 0
    for r in recs:
        try:
            e = enc.encode(r)
        except ValueError:
            fail += 1
            continue
        if e is None:
            continue
        parts = [e["c"], e["r"]] if "c" in e else [e]
        n_ok += 1
        for p in parts:
            tgt += p["n_target"]; tot += len(p["input_ids"]); mx = max(mx, len(p["input_ids"]))
    return {"n": len(recs), "n_encoded": n_ok, "target_tokens": tgt, "total_tokens": tot, "max_len": mx, "prefix_failures": fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--sample", type=int, default=0, help="encode a random sample per arm and extrapolate (fast)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tasks, by_id, tools_fn = load_context()
    out = {"tokenizer": a.tokenizer, "max_len": a.max_len, "arms": {}}
    for arm in ARMS:
        if ARMS[arm]["objective"] is None:
            continue
        enc = Encoder(HFRenderer(tok), tools_fn, a.max_len)
        recs, anchors, _ = build_records(arm, tasks, by_id, 0)
        n_all = (len(recs), len(anchors))
        if a.sample:
            import random
            rng = random.Random(0)
            recs = rng.sample(recs, min(a.sample, len(recs))); anchors = rng.sample(anchors, min(a.sample, len(anchors)))
        s = {"primary": stats(enc, recs), "anchor": stats(enc, anchors) if anchors else None, "n_all": n_all}
        if a.sample:
            for k, n in (("primary", n_all[0]), ("anchor", n_all[1])):
                if s[k] and s[k]["n_encoded"]:
                    f = n / s[k]["n"]
                    s[k]["extrapolated_total_tokens"] = round(s[k]["total_tokens"] * f)
                    s[k]["mean_tokens_per_seq"] = round(s[k]["total_tokens"] / (s[k]["n_encoded"] * (2 if "pair" in str(arm) or ARMS[arm]["objective"] == "dpo" and k == "primary" else 1)))
        s["primary"]["dropped_by_length"] = s["primary"]["n"] - s["primary"]["n_encoded"] - s["primary"]["prefix_failures"]
        out["arms"][arm] = s
        print(arm, json.dumps(s), flush=True)
    enc = Encoder(HFRenderer(tok), tools_fn, a.max_len)
    val = build_val()
    out["dev_val_sets"] = {k: len(v) for k, v in val.items()}
    od = os.path.join(ROOT, "results", "budget"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "arm_budgets.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
