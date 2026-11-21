"""Summarise baseline runs: success (Wilson CI), cost per success, waste, latency.

Usage: python scripts/summarize_runs.py results/pod_smoke results/pod_fewshot
Prints a markdown table; one row per <run>/<policy> directory.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dare.evaluation.metrics import wilson_interval  # noqa: E402
from dare.utils.logging import load_run  # noqa: E402


def main(dirs):
    print("| run | policy | n | success | 95% CI | env actions / success | LM tokens / success | "
          "wasted rate | LM s / step | wall s / episode |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        for pol in sorted(os.listdir(d)):
            p = os.path.join(d, pol)
            if not os.path.isfile(os.path.join(p, "episodes.jsonl")):
                continue
            eps, trs = load_run(p)
            n, k = len(eps), sum(e["success"] for e in eps)
            lo, hi = wilson_interval(k, n)
            steps = sum(e["n_steps"] for e in eps)
            toks = sum(e["tokens_total"] for e in eps)
            lm_s = sum(t["time_s"]["lm"] + t["time_s"]["select"] for t in trs) / max(len(trs), 1)
            per = (lambda x: f"{x / k:.0f}" if k else "∞")
            print(f"| {os.path.basename(d)} | {pol} | {n} | {k}/{n} | {lo:.2f}–{hi:.2f} | {per(steps)} | {per(toks)} | "
                  f"{sum(e['n_wasted'] for e in eps) / max(steps, 1):.2f} | {lm_s:.3f} | "
                  f"{sum(e['wall_s'] for e in eps) / max(n, 1):.1f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
