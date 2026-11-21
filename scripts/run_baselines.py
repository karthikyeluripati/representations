"""Run baseline policies on ALFWorld games with full transition logging; resumable.

Usage:  python scripts/run_baselines.py --config configs/baseline.yaml [--policies random frozen_lm]
            [--n-games 5] [--split valid_seen] [--out results/smoke] [--verify]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dare.agents.demos import build_demos  # noqa: E402
from dare.agents.policies import make_policy  # noqa: E402
from dare.environments.alfworld_env import AlfWorldEnv, list_games, select_games  # noqa: E402
from dare.evaluation.metrics import wilson_interval  # noqa: E402
from dare.evaluation.runner import episode_id_for, run_episode  # noqa: E402
from dare.evaluation.verify import verify_episodes  # noqa: E402
from dare.utils.common import config_hash, load_config, seed_everything  # noqa: E402
from dare.utils.logging import RunLogger, load_run  # noqa: E402


def _model_key(cfg):
    """Full model config for the config hash. Batch size, device and prefix reuse all change bf16
    numerics and hence greedy decisions (observed), so none of them is treated as performance-only."""
    return dict(cfg.get("model") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default="configs/environment.yaml")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--policies", nargs="*")
    ap.add_argument("--n-games", type=int)
    ap.add_argument("--split")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--out")
    ap.add_argument("--max-steps", type=int, help="override env.max_steps (e.g. tiny CPU smoke tests)")
    ap.add_argument("--verify", action="store_true", help="replay every logged episode and check consistency")
    args = ap.parse_args()

    cfg = load_config(args.env_config, args.config)
    run = cfg["run"]
    for k in ("policies", "n_games", "split", "seeds", "out"):
        v = getattr(args, k)
        if v is not None:
            run["out_dir" if k == "out" else k] = v
    ecfg = cfg["env"]
    if args.max_steps is not None:
        ecfg["max_steps"] = args.max_steps  # part of the config hash, so it cannot silently mix with other runs
    games = select_games(list_games(ecfg["data_root"], run["split"], ecfg.get("task_families")), run["n_games"])
    seed_everything(0)
    scorer = None
    if any(p != "random" for p in run["policies"]):
        from dare.agents.lm import LMScorer

        m = cfg["model"]
        t0 = time.perf_counter()
        scorer = LMScorer(m["id"], m.get("dtype", "float32"), m.get("num_threads"), m.get("max_batch", 64),
                          m.get("device", "auto"), m.get("reuse_prefix", True))
        print(f"loaded {m['id']} in {time.perf_counter() - t0:.1f}s")
    env = AlfWorldEnv(max_steps=ecfg["max_steps"])
    for pname in run["policies"]:
        pcfg = {k: v for k, v in cfg.items()}
        h = config_hash({"env": ecfg, "model": _model_key(cfg) if pname != "random" else None,
                         "policy": cfg.get("policy") if pname != "random" else None, "policy_name": pname})
        out = os.path.join(run["out_dir"], pname)
        logger = RunLogger(out, meta={"policy": pname, "config_hash": h, "config": pcfg, "n_games": len(games),
                                      "model_id": scorer.model_id if (scorer and pname != "random") else None})
        done = logger.completed_episode_ids()
        demos = None
        if cfg.get("policy", {}).get("few_shot") and pname != "random":
            demos = build_demos(ecfg["data_root"], cfg["policy"].get("demo_cache", "results/demos/demos_v1.json"))
        policy = make_policy(pname, scorer, cfg, demos)
        for seed in run["seeds"]:
            for g in games:
                if episode_id_for(run["out_dir"], pname, g.task_id, seed) in done:
                    continue
                s = run_episode(env, policy, g, seed, run["out_dir"], logger, ecfg.get("max_candidates"),
                                run.get("token_budget"), scorer.model_id if (scorer and policy.uses_lm) else None, h)
                print(f"[{pname}] {g.task_id} seed={seed} success={s['success']} steps={s['n_steps']} "
                      f"wasted={s['n_wasted']} tokens={s['tokens_total']} wall={s['wall_s']:.1f}s", flush=True)
        eps, trs = load_run(out)
        k, n = sum(e["success"] for e in eps), len(eps)
        lo, hi = wilson_interval(k, n)
        print(f"== {pname}: success {k}/{n} (95% Wilson CI {lo:.2f}-{hi:.2f}), "
              f"mean steps {sum(e['n_steps'] for e in eps) / max(n, 1):.1f}, "
              f"wasted-action rate {sum(e['n_wasted'] for e in eps) / max(sum(e['n_steps'] for e in eps), 1):.2f}")
        if args.verify:
            rep = verify_episodes(eps, trs, ecfg["max_steps"])
            print(f"== {pname}: VERIFY {'PASS' if rep['ok'] else 'FAIL'} "
                  f"({rep['n_episodes']} episodes, {rep['n_steps']} steps, {len(rep['mismatches'])} mismatches)")
            with open(os.path.join(out, "verify.json"), "w") as f:
                json.dump(rep, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
