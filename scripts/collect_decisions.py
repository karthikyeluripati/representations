"""Phase 2: build a counterfactual decision dataset from TRAIN-split trajectories.

For each source trajectory, sample decision states; at each state take K
candidate actions (the source policy's own choice + K-1 uniformly sampled
admissible alternatives — NOT "all actions on a successful path are good"),
restore the state by verified replay, execute each candidate, and continue with
the same frozen rollout policy for n_rollouts stochastic rollouts.

Output (resumable): <out>/decisions.jsonl, one row per (state, candidate);
<out>/emb/<state_hash>.npz with the frozen-LM context embedding and the
per-candidate embeddings (if an LM is loaded).

Usage:
  python scripts/collect_decisions.py                                   # LM source + LM rollouts (GPU)
  python scripts/collect_decisions.py --source-policy random --rollout-policy random --out results/decisions_proxy
      # CPU PROXY ONLY: random-policy branch values; not a language-model result.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from dare.agents.lm import build_prompt_messages, static_user_prefix  # noqa: E402
from dare.agents.demos import build_demos  # noqa: E402
from dare.agents.policies import make_policy  # noqa: E402
from dare.datasets.branching import estimate_branch, restore  # noqa: E402
from dare.datasets.splits import split_by_task  # noqa: E402
from dare.environments.alfworld_env import AlfWorldEnv, list_games, select_games  # noqa: E402
from dare.evaluation.runner import episode_id_for, run_episode  # noqa: E402
from dare.utils.common import config_hash, derive_seed, load_config, stable_hash  # noqa: E402
from dare.utils.logging import RunLogger, load_run, read_jsonl  # noqa: E402


def _model_key(cfg):
    """Full model config for the config hash. Batch size, device and prefix reuse all change bf16
    numerics and hence greedy decisions (observed), so none of them is treated as performance-only."""
    return dict(cfg.get("model") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default="configs/environment.yaml")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--rep-config", default="configs/representation.yaml")
    ap.add_argument("--source-policy", default="frozen_lm")
    ap.add_argument("--rollout-policy")
    ap.add_argument("--n-games", type=int)
    ap.add_argument("--n-rollouts", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--out")
    ap.add_argument("--shard", default="0/1", help="i/n: process games[i::n]; each shard writes its own files")
    ap.add_argument("--embed", action="store_true", help="load the LM only to embed decision states (proxy runs)")
    args = ap.parse_args()
    cfg = load_config(args.env_config, args.config, args.rep_config)
    dc, ecfg = cfg["decisions"], cfg["env"]
    for k, ck in (("rollout_policy", "rollout_policy"), ("n_games", "n_games"), ("n_rollouts", "n_rollouts"),
                  ("out", "out_dir")):
        if getattr(args, k) is not None:
            dc[ck] = getattr(args, k)
    if args.max_steps is not None:
        ecfg["max_steps"] = args.max_steps
    out = dc["out_dir"]
    os.makedirs(os.path.join(out, "emb"), exist_ok=True)
    if dc["split"] != "train":
        raise SystemExit("decision states must come from the train split (leakage guard)")

    needs_lm = args.source_policy != "random" or dc["rollout_policy"] != "random" or args.embed
    scorer = None
    if needs_lm:
        from dare.agents.lm import LMScorer

        m = cfg["model"]
        scorer = LMScorer(m["id"], m.get("dtype", "float32"), m.get("num_threads"), m.get("max_batch", 16),
                          m.get("device", "auto"), m.get("reuse_prefix", True))
    demos = None
    if cfg.get("policy", {}).get("few_shot"):
        demo_cache = cfg["policy"].get("demo_cache", "results/demos/demos_v1.json")
        demos = build_demos(ecfg["data_root"], demo_cache)
        with open(demo_cache + ".sources.json") as f:
            demo_tasks = set(json.load(f).values())
    else:
        demo_tasks = set()
    src_policy = make_policy(args.source_policy, scorer, cfg, demos)
    roll_cfg = {**cfg, "policy": {**cfg.get("policy", {}), "temperature": dc["rollout_temperature"]}}
    roll_policy = make_policy(dc["rollout_policy"], scorer, roll_cfg, demos)

    pool = [g for g in list_games(ecfg["data_root"], "train", ecfg.get("task_families")) if g.task_id not in demo_tasks]
    games = select_games(pool, dc["n_games"])  # demo-source tasks excluded (leakage guard)
    si, sn = map(int, args.shard.split("/"))
    games = games[si::sn]
    tag = "" if sn == 1 else f"_shard{si}of{sn}"
    assign = split_by_task([g.task_id for g in games], dc["dev_fraction"],
                           holdout_families=tuple(dc.get("holdout_families", ())))
    h = config_hash({"env": ecfg, "model": _model_key(cfg) if needs_lm else None, "policy": cfg.get("policy"),
                     "decisions": dc, "source": args.source_policy})
    env = AlfWorldEnv(max_steps=ecfg["max_steps"])
    # bf16 numerics (hence greedy decisions) differ across GPU models: record where each row was produced
    hw = "cpu"
    if scorer is not None and str(scorer.device).startswith("cuda"):
        import torch

        hw = torch.cuda.get_device_name(0)

    # 1) source trajectories (logged + verified-by-replay later)
    src_dir = os.path.join(out, "source" + tag)
    lg = RunLogger(src_dir, meta={"config_hash": h, "policy": args.source_policy})
    done = lg.completed_episode_ids()
    for g in games:
        if episode_id_for(src_dir, src_policy.name, g.task_id, 0) not in done:
            run_episode(env, src_policy, g, 0, src_dir, lg, ecfg.get("max_candidates"), None,
                        scorer.model_id if scorer else None, h)
    eps, trs = load_run(src_dir)
    by_ep: dict[str, list[dict]] = {}
    for t in trs:
        by_ep.setdefault(t["episode_id"], []).append(t)
    game_of = {g.task_id: g for g in games}

    # 2) branch each sampled decision state
    dpath = os.path.join(out, f"decisions{tag}.jsonl")
    have = {(r["state_id"], r["candidate_action"]) for r in read_jsonl(dpath)}
    n_rows, t0 = 0, time.perf_counter()
    for e in eps:
        g = game_of[e["task_id"]]
        steps = sorted(by_ep[e["episode_id"]], key=lambda t: t["step"])
        eligible = [t for t in steps if len(t["candidates"]) >= 2]
        rng = random.Random(derive_seed("states", e["episode_id"]))
        for t in sorted(rng.sample(eligible, min(dc["states_per_episode"], len(eligible))), key=lambda t: t["step"]):
            prefix = e["actions"][: t["step"]]
            state_id = stable_hash(g.task_id, *prefix)
            full_adm = t["candidates"] if ecfg.get("max_candidates") is None else None
            st = restore(env, g, prefix, t["state_fp"], t["observation"], full_adm)
            others = [c for c in t["candidates"] if c != t["selected_action"]]
            cands = [t["selected_action"]] + rng.sample(others, min(dc["candidates_per_state"] - 1, len(others)))
            if scorer is not None and not os.path.exists(os.path.join(out, "emb", f"{state_id}.npz")):
                hist = list(zip(prefix, [s["next_observation"] for s in steps[: t["step"]]]))
                intro = steps[0]["observation"]
                demo = (demos or {}).get(g.task_family)
                r = scorer.score(build_prompt_messages(intro, hist, st.observation, t["candidates"],
                                                       cfg["policy"].get("max_history", 8), demo), cands,
                                 static_user_prefix(intro, demo))
                np.savez(os.path.join(out, "emb", f"{state_id}.npz"), context=r.context_emb, actions=r.action_emb,
                         candidates=np.array(cands), logp_mean=r.logp_mean, logp_sum=r.logp_sum)
            for c in cands:
                if (state_id, c) in have:
                    continue
                b = estimate_branch(env, roll_policy, g, prefix, t["state_fp"], c, dc["n_rollouts"],
                                    derive_seed("branch", h), ecfg.get("max_candidates"))
                row = {"task_id": g.task_id, "task_family": g.task_family, "fit_split": assign[g.task_id],
                       "state_id": state_id, "step": t["step"], "objective": t["objective"],
                       "history_ref": {"episode_id": e["episode_id"], "n_prev_steps": t["step"]},
                       "observation": t["observation"], "state_fp": t["state_fp"],
                       "is_policy_choice": c == t["selected_action"], "n_admissible": len(t["candidates"]),
                       "source_policy": args.source_policy, "rollout_policy": roll_policy.name,
                       "rollout_temperature": dc["rollout_temperature"] if roll_policy.uses_lm else None,
                       "proxy": roll_policy.name == "random", "config_hash": h, "hardware": hw,
                       "representation_file": f"emb/{state_id}.npz" if scorer else None, **b}
                with open(dpath, "a") as f:
                    f.write(json.dumps(row) + "\n")
                n_rows += 1
                print(f"{g.task_id[:50]} step={t['step']} cand={c!r} succ={b['branch_success_rate']:.2f} "
                      f"rem={b['estimated_remaining_actions']:.1f}", flush=True)
    env.close()
    print(f"wrote {n_rows} new rows in {time.perf_counter() - t0:.0f}s -> {dpath}")


if __name__ == "__main__":
    main()
