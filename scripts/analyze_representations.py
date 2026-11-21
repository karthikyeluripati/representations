"""Phase 3 (pre-training sanity analysis): do untrained signals rank candidates by branch outcome?

NO model is fit here. For each decision state with >=2 candidates and
non-constant branch outcomes, compare within-state rankings induced by:
  * lm_logp_mean        frozen-LM mean token log-likelihood of the candidate
  * novelty_centroid    cosine distance of the candidate embedding from the mean
                        candidate embedding at that state (representation novelty)
against branch_success_rate (and -estimated_remaining_actions).
Reports pairwise ranking accuracy and mean within-state Spearman, with 95%
bootstrap CIs clustered by task. Learned projections / value heads (Phase 3
proper) are deliberately not implemented yet.

Usage: python scripts/analyze_representations.py --decisions results/decisions_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from dare.evaluation.metrics import cluster_bootstrap_ci, pairwise_ranking_accuracy, spearman  # noqa: E402
from dare.utils.logging import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--subset", default="fit", help="fit | dev | family_holdout | all")
    args = ap.parse_args()
    import glob

    rows = [r for f in sorted(glob.glob(os.path.join(args.decisions, "decisions*.jsonl"))) for r in read_jsonl(f)]
    if args.subset != "all":
        rows = [r for r in rows if r["fit_split"] == args.subset]
    if not rows:
        raise SystemExit("no rows")
    if any(r.get("proxy") for r in rows):
        print("WARNING: branch values come from a RANDOM rollout policy (proxy); not LM-policy values.")
    by_state = defaultdict(list)
    for r in rows:
        by_state[r["state_id"]].append(r)
    feats = {"lm_logp_mean": [], "novelty_centroid": []}
    target_succ, target_cost, groups, tasks = [], [], [], []
    n_states_used = 0
    for sid, rs in by_state.items():
        ef = os.path.join(args.decisions, rs[0]["representation_file"] or "")
        if not rs[0]["representation_file"] or not os.path.exists(ef):
            continue
        z = np.load(ef)
        cand = list(z["candidates"])
        E = z["actions"] / (np.linalg.norm(z["actions"], axis=1, keepdims=True) + 1e-8)
        cen = E.mean(0)
        cen /= np.linalg.norm(cen) + 1e-8
        nov = 1 - E @ cen
        n_states_used += 1
        for r in rs:
            i = cand.index(r["candidate_action"])
            feats["lm_logp_mean"].append(float(z["logp_mean"][i]))
            feats["novelty_centroid"].append(float(nov[i]))
            target_succ.append(r["branch_success_rate"])
            target_cost.append(-r["estimated_remaining_actions"])
            groups.append(sid)
            tasks.append(r["task_id"])
    if not groups:
        raise SystemExit("no rows with representation files (was an LM loaded during collection?)")
    groups, tasks = np.array(groups), np.array(tasks)
    ys = {"success": np.array(target_succ), "neg_remaining_actions": np.array(target_cost)}
    informative = {sid for sid in set(groups) if len(set(ys["success"][groups == sid])) > 1}
    report = {"n_rows": len(groups), "n_states": n_states_used, "n_tasks": len(set(tasks)),
              "n_states_with_outcome_variation": len(informative), "results": {}}
    for fname, f in feats.items():
        f = np.array(f)
        for yname, y in ys.items():
            def pra(idx):
                return pairwise_ranking_accuracy(f[idx], y[idx], groups[idx])

            def msp(idx):
                vals = [spearman(f[idx][groups[idx] == g], y[idx][groups[idx] == g]) for g in set(groups[idx])]
                vals = [v for v in vals if not np.isnan(v)]
                return float(np.mean(vals)) if vals else float("nan")

            report["results"][f"{fname}->{yname}"] = {
                "pairwise_acc": cluster_bootstrap_ci(pra, tasks, n_boot=1000),
                "mean_within_state_spearman": cluster_bootstrap_ci(msp, tasks, n_boot=1000)}
    print(json.dumps(report, indent=2))
    with open(os.path.join(args.decisions, f"analysis_{args.subset}.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
