"""Phase A audit of a branch-outcome decision dataset (read-only).

Usage: python scripts/audit_decisions.py --decisions results/decisions_v1_3b [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from dare.datasets.splits import split_by_task  # noqa: E402
from dare.utils.logging import load_run, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--json")
    ap.add_argument("--demo-sources", default="results/demos/demos_v1.json.sources.json")
    args = ap.parse_args()
    D = args.decisions
    rows = [r for f in sorted(glob.glob(os.path.join(D, "decisions*.jsonl"))) for r in read_jsonl(f)]
    rep = {"n_rows": len(rows)}
    issues = []

    # ---- duplicates / config
    keys = collections.Counter((r["state_id"], r["candidate_action"]) for r in rows)
    rep["duplicate_rows"] = sum(v > 1 for v in keys.values())
    rep["config_hashes"] = sorted({r["config_hash"] for r in rows})
    rep["hardware"] = dict(collections.Counter(r["hardware"] for r in rows))
    rep["rollout_counts"] = dict(collections.Counter(r["rollout_count"] for r in rows))
    rep["rollout_policy"] = sorted({(r["rollout_policy"], r["rollout_temperature"]) for r in rows})

    # ---- split integrity (recompute from task ids; must match stored labels)
    task_split = collections.defaultdict(set)
    for r in rows:
        task_split[r["task_id"]].add(r["fit_split"])
    multi = [t for t, s in task_split.items() if len(s) > 1]
    if multi:
        issues.append(f"{len(multi)} tasks with >1 split label")
    recomputed = split_by_task(list(task_split), 0.25, holdout_families=("pick_two_obj_and_place",))
    mism = [t for t, s in task_split.items() if recomputed[t] not in s]
    if mism:
        issues.append(f"{len(mism)} tasks whose stored split != recomputed split")
    state_split = collections.defaultdict(set)
    for r in rows:
        state_split[r["state_id"]].add(r["fit_split"])
    rep["states_in_multiple_splits"] = sum(len(s) > 1 for s in state_split.values())
    # same scene (room/layout) across splits: task id = family-obj-mrecep-recep-sceneNum/trial
    scene = lambda t: t.split("/")[0].split("-")[-1]  # noqa: E731
    sc_split = collections.defaultdict(set)
    for t, s in task_split.items():
        sc_split[scene(t)] |= s
    rep["scenes_shared_across_splits"] = sorted(k for k, s in sc_split.items() if len(s) > 1)
    base = lambda t: t.split("/")[0]  # noqa: E731  (same goal+scene, different trial)
    b_split = collections.defaultdict(set)
    for t, s in task_split.items():
        b_split[base(t)] |= s
    rep["goal_scene_pairs_shared_across_splits"] = sorted(k for k, s in b_split.items() if len(s) > 1)
    if os.path.exists(args.demo_sources):
        demo_tasks = set(json.load(open(args.demo_sources)).values())
        rep["demo_source_tasks_in_data"] = sorted(demo_tasks & set(task_split))

    # ---- states: prefixes, informativeness
    by_state = collections.defaultdict(list)
    for r in rows:
        by_state[r["state_id"]].append(r)
    states = []
    for sid, rs in by_state.items():
        sr = [r["branch_success_rate"] for r in rs]
        ra = [r["estimated_remaining_actions"] for r in rs]
        # a pair is "reliable" if Beta posteriors' means differ by > their combined std (≈1-sigma separation)
        rel = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                d = abs(rs[i]["success_posterior_mean"] - rs[j]["success_posterior_mean"])
                if d > np.hypot(rs[i]["outcome_uncertainty"], rs[j]["outcome_uncertainty"]):
                    rel += 1
        states.append({"state_id": sid, "task_id": rs[0]["task_id"], "family": rs[0]["task_family"],
                       "split": rs[0]["fit_split"], "step": rs[0]["step"], "n_cand": len(rs),
                       "succ_varies": len(set(sr)) > 1, "rem_varies": len(set(ra)) > 1,
                       "reliable_pairs": rel, "mean_succ": float(np.mean(sr)),
                       "policy_choice_present": sum(r["is_policy_choice"] for r in rs) == 1,
                       "episode": rs[0]["history_ref"]["episode_id"]})
    rep["n_states"] = len(states)
    rep["candidates_per_state"] = dict(collections.Counter(s["n_cand"] for s in states))
    rep["states_without_exactly_one_policy_choice"] = sum(not s["policy_choice_present"] for s in states)
    # shared trajectory prefixes: states from the same source episode share prefixes (nested)
    per_ep = collections.Counter(s["episode"] for s in states)
    rep["states_per_source_episode"] = dict(collections.Counter(per_ep.values()))
    fp_per_task = collections.defaultdict(collections.Counter)
    for r in rows:
        fp_per_task[r["task_id"]][r["state_fp"]] += 0
    for s in states:
        fp_per_task[s["task_id"]][by_state[s["state_id"]][0]["state_fp"]] += 1
    rep["same_symbolic_state_sampled_twice_in_task"] = sum(any(v > 1 for v in c.values()) for c in fp_per_task.values())

    # ---- label properties
    rep["labels_are"] = "Monte-Carlo estimates: success rate over n rollouts of the frozen rollout policy"
    all_out = [o for r in rows for o in r["rollout_outcomes"]]
    rep["overall_rollout_success"] = float(np.mean(all_out))
    succ_rates = collections.Counter(round(r["branch_success_rate"], 3) for r in rows)
    rep["branch_success_rate_hist"] = dict(sorted(succ_rates.items()))
    rep["mean_outcome_uncertainty"] = float(np.mean([r["outcome_uncertainty"] for r in rows]))
    # identical rollout outcome vectors across candidates in a state (possible sign of shared seeds/correlation)
    same_vec = sum(len({tuple(r["rollout_outcomes"]) for r in rs}) < len(rs) for rs in by_state.values())
    rep["states_with_identical_outcome_vectors_across_candidates"] = same_vec
    # remaining-actions censoring: failures run to the step limit, so remaining actions is partly a success proxy
    rep["corr_success_vs_remaining_actions"] = float(np.corrcoef([r["branch_success_rate"] for r in rows],
                                                                 [r["estimated_remaining_actions"] for r in rows])[0, 1])

    # ---- source trajectories
    eps = []
    for sd in sorted(glob.glob(os.path.join(D, "source*"))):
        e, _ = load_run(sd)
        eps += e
    rep["source_episodes"] = len(eps)
    rep["source_success"] = int(sum(e["success"] for e in eps))

    # ---- embeddings
    bad, nan, mism_c, mism_lp, choice_not_argmax = [], 0, 0, 0, 0
    dims = collections.Counter()
    for sid, rs in by_state.items():
        p = os.path.join(D, rs[0]["representation_file"] or "")
        try:
            z = np.load(p)
            ctx, act, cands, lp = z["context"], z["actions"], list(z["candidates"]), z["logp_mean"]
        except Exception as ex:  # noqa: BLE001
            bad.append((sid, str(ex)[:60]))
            continue
        dims[(ctx.shape, act.shape)] += 1
        if not (np.isfinite(ctx).all() and np.isfinite(act).all() and np.isfinite(lp).all()):
            nan += 1
        if sorted(cands) != sorted(r["candidate_action"] for r in rs):
            mism_c += 1
        pc = [r["candidate_action"] for r in rs if r["is_policy_choice"]]
        if pc and cands[int(np.argmax(lp))] != pc[0]:
            choice_not_argmax += 1
    rep["embedding_files_bad"] = bad
    rep["embedding_nonfinite"] = nan
    rep["embedding_candidate_mismatch"] = mism_c
    rep["embedding_shapes"] = {str(k): v for k, v in dims.items()}
    rep["policy_choice_not_argmax_of_stored_logp"] = choice_not_argmax
    rep["embedding_future_info"] = ("none by construction: computed from the decision-time prompt "
                                    "(intro, <=8 history steps, current obs, admissible list) before any rollout")

    # ---- effective sample sizes
    def ess(sub):
        inf = [s for s in sub if s["succ_varies"]]
        return {"tasks": len({s["task_id"] for s in sub}), "states": len(sub),
                "informative_states": len(inf), "tasks_with_informative": len({s["task_id"] for s in inf}),
                "reliable_pairs": int(sum(s["reliable_pairs"] for s in sub)),
                "states_with_reliable_pair": sum(s["reliable_pairs"] > 0 for s in sub),
                "rem_varying_states": sum(s["rem_varies"] for s in sub)}
    rep["by_split"] = {sp: ess([s for s in states if s["split"] == sp]) for sp in ("fit", "dev", "family_holdout")}
    rep["by_family"] = {f: {**ess([s for s in states if s["family"] == f]),
                            "splits": sorted({s["split"] for s in states if s["family"] == f})}
                        for f in sorted({s["family"] for s in states})}
    inf_per_task = collections.Counter(s["task_id"] for s in states if s["succ_varies"])
    rep["informative_states_per_task"] = dict(collections.Counter(inf_per_task.values()))
    rep["issues"] = issues
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if args.json:
        open(args.json, "w").write(txt)


if __name__ == "__main__":
    main()
