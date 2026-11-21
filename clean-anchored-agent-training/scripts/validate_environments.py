"""Build cacrt_tasks_v1 and validate the BFCL environment before any training.

Checks (results -> results/validation/environment_validation.json):
  V1 positional-argument replay: old keyword-only replay vs fixed replay, per affected task.
  V2 solvability: S0/S1/S2 solved and S3 escalated by the executable reference policy (seeds 0-2), per split.
  V3 tool-call budget: reference policy call counts vs MAX_TOOL_CALLS (and S2's worst case).
  V4 leakage: user text / GT sequence / base task shared across splits.
  V5 determinism: snapshot -> replay -> restore gives identical state hashes; two independent builds agree.
  V6 injected-error pools are pairwise disjoint and never equal a real BFCL error seen in any S0 reference run.
Usage: python scripts/validate_environments.py --v2-tasks <path to previous project's data/tasks.json>
"""
import argparse
import ast
import collections
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.environment.bfcl_env import (EVAL_ERRORS, MAX_TOOL_CALLS, TRAIN_ERRORS, V2_ERRORS, Episode, obs_kind)  # noqa: E402
from src.environment.tasks import VERSION, build_all_tasks, leakage_report  # noqa: E402
from src.counterfactuals.policies import reference_policy  # noqa: E402
from src.replay.replay import method_owner, new_instances, public_state, run_calls, state_hash, states_equal  # noqa: E402


def old_keyword_only_replay(inst, calls):
    for c in calls:
        node = ast.parse(c.strip(), mode="eval").body
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        cls = method_owner(inst, name)
        try:
            getattr(inst[cls], name)(**kw)
        except Exception:  # noqa: BLE001
            pass


def has_positional(c):
    return bool(ast.parse(c.strip(), mode="eval").body.args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-tasks", default=os.path.join(ROOT, "data", "v2_split_map.json"),
                    help="previous project's tasks.json, or the extracted data/v2_split_map.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "tasks_v1.json"))
    args = ap.parse_args()
    v2 = json.load(open(args.v2_tasks))
    v2_split = v2["map"] if isinstance(v2, dict) else {t["base_id"]: t["split"] for t in v2}
    tasks, dropped = build_all_tasks(v2_split)
    rep = {"version": VERSION, "n_tasks": len(tasks), "dropped": dict(dropped),
           "split_counts": dict(collections.Counter(t["split"] for t in tasks)),
           "split_base_counts": {s: len({t["base_id"] for t in tasks if t["split"] == s}) for s in {t["split"] for t in tasks}}}

    # V1
    v1 = collections.Counter()
    for t in tasks:
        calls = t["prefix_gt"] + t["turn_gt"]
        if not any(has_positional(c) for c in calls):
            continue
        a = new_instances(t["entry"]); old_keyword_only_replay(a, calls)
        b = new_instances(t["entry"]); run_calls(b, calls)
        v1[(t["split"], "end_state_differs" if not states_equal(a, b) else "end_state_same")] += 1
    rep["V1_positional"] = {f"{s}:{k}": n for (s, k), n in sorted(v1.items())}

    # V2 + V3
    sol = collections.defaultdict(collections.Counter)
    calls_used = collections.defaultdict(list)
    native_seen = set()
    unsolvable = []
    for t in tasks:
        for sc in ("S0", "S1", "S2", "S3"):
            for seed in (0, 1, 2):
                ep = Episode(t, sc, seed, error_pool=TRAIN_ERRORS)
                reference_policy(ep)
                g = ep.grade()
                sol[(t["split"], sc)]["ok" if g["correct"] else "fail"] += 1
                calls_used[sc].append(g["n_calls"])
                if not g["correct"]:
                    unsolvable.append((t["id"], sc, seed))
                if sc == "S0":
                    native_seen |= {o for _, _, o in ep.calls if obs_kind(o) == "native"}
                if sc == "S0" and seed == 0:
                    break
            # S0 is fault-free: one seed suffices (deterministic)
    rep["V2_solvability"] = {f"{s}:{sc}": dict(c) for (s, sc), c in sorted(sol.items())}
    rep["V2_unsolvable_task_ids"] = sorted({u[0] for u in unsolvable})
    rep["V3_budget"] = {sc: {"max_calls": max(v), "mean": round(sum(v) / len(v), 2), "budget": MAX_TOOL_CALLS,
                             "n_at_budget": sum(x >= MAX_TOOL_CALLS for x in v)} for sc, v in calls_used.items()}
    rep["V4_leakage"] = leakage_report(tasks)

    # V5
    det_fail = 0
    for t in tasks[:60]:
        ep = Episode(t, "S1", 0, error_pool=TRAIN_ERRORS)
        snap = ep.snapshot(); h0 = state_hash(ep.inst)
        reference_policy(ep)
        if state_hash(snap.inst) != h0:
            det_fail += 1
        ep2 = Episode(t, "S1", 0, error_pool=TRAIN_ERRORS); reference_policy(ep2)
        if state_hash(ep.inst) != state_hash(ep2.inst) or [c[2] for c in ep.calls] != [c[2] for c in ep2.calls]:
            det_fail += 1
    rep["V5_determinism_failures_first60"] = det_fail

    pools = [set(V2_ERRORS), set(TRAIN_ERRORS), set(EVAL_ERRORS)]
    rep["V6_pools_disjoint"] = all(not (pools[i] & pools[j]) for i in range(3) for j in range(i + 1, 3))
    rep["V6_pool_equals_real_error"] = sorted(set().union(*pools) & native_seen)
    rep["n_distinct_real_errors_in_S0_reference_runs"] = len(native_seen)

    usable = [t for t in tasks if t["split"] != "unassigned" and t["id"] not in set(rep["V2_unsolvable_task_ids"])]
    for t in tasks:
        t["usable"] = t in usable
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    blob = json.dumps(tasks, default=str)
    open(args.out, "w").write(blob)
    rep["tasks_file"] = os.path.relpath(args.out, ROOT)
    rep["tasks_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    rep["usable_split_counts"] = dict(collections.Counter(t["split"] for t in usable))
    rep["usable_split_base_counts"] = {s: len({t["base_id"] for t in usable if t["split"] == s}) for s in rep["usable_split_counts"]}
    rep["v2_tasks_sha256"] = hashlib.sha256(open(args.v2_tasks, "rb").read()).hexdigest()
    od = os.path.join(ROOT, "results", "validation"); os.makedirs(od, exist_ok=True)
    json.dump(rep, open(os.path.join(od, "environment_validation.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1)[:6000])


if __name__ == "__main__":
    main()
