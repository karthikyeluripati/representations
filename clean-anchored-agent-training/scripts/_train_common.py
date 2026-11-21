"""Shared CLI for the three training entry points."""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.training.run import add_args, run_arm  # noqa: E402


def main(allowed, doc):
    ap = add_args(argparse.ArgumentParser(description=doc))
    ap.add_argument("--arm", required=True, choices=allowed)
    a = ap.parse_args()
    cfg, summ, hist = run_arm(a.arm, a)
    last = hist[-1] if hist else {}
    print(json.dumps({"arm": a.arm, "steps": cfg["steps"], "n_primary": summ["n_primary"], "n_anchor": summ["n_anchor"],
                      "last": {k: v for k, v in last.items() if k != "decision"}, "decision": next((h["decision"] for h in reversed(hist) if "decision" in h), None)}))
