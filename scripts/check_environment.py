"""Report hardware/software and verify ALFWorld + deterministic replay.

Usage (inside WSL/Linux venv):  python scripts/check_environment.py [--episodes 3]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dare.environments.alfworld_env import AlfWorldEnv, list_games  # noqa: E402
from dare.utils.common import load_config  # noqa: E402


def system_report() -> dict:
    rep = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        with open("/proc/meminfo") as f:
            rep["mem_total_gb"] = round(int(f.readline().split()[1]) / 1e6, 2)
    except OSError:
        pass
    rep["disk_free_gb"] = round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9, 1)
    try:
        import torch

        rep.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        rep["torch"] = None
    for m in ("textworld", "alfworld", "transformers"):
        try:
            mod = __import__(m)
            rep[m] = getattr(mod, "__version__", "installed")
        except ImportError:
            rep[m] = None
    return rep


def replay_check(env: AlfWorldEnv, game_file: str, n_steps: int, seed: int) -> dict:
    """Random walk, then replay the same actions in a fresh env; compare every step."""
    rng = random.Random(seed)
    st = env.reset(game_file)
    trace = [(st.fingerprint, st.observation, tuple(st.admissible))]
    acts = []
    while not st.done and len(acts) < n_steps:
        a = rng.choice(st.admissible)
        acts.append(a)
        st = env.step(a)
        trace.append((st.fingerprint, st.observation, tuple(st.admissible)))
    fresh = AlfWorldEnv(max_steps=env.max_steps)
    st2 = fresh.reset(game_file)
    trace2 = [(st2.fingerprint, st2.observation, tuple(st2.admissible))]
    for a in acts:
        st2 = fresh.step(a)
        trace2.append((st2.fingerprint, st2.observation, tuple(st2.admissible)))
    fresh.close()
    # branch test: from the midpoint, two different actions must give different / reproducible states
    mid = len(acts) // 2
    s_mid = env.replay(game_file, acts[:mid])
    alts = s_mid.admissible[:2]
    outs = []
    for a in alts:
        for _ in range(2):
            env.replay(game_file, acts[:mid])
            outs.append((a, env.step(a).fingerprint))
    reproducible = all(outs[2 * i][1] == outs[2 * i + 1][1] for i in range(len(alts)))
    return {"n_steps": len(acts),
            "fingerprint_equal": all(x[0] == y[0] for x, y in zip(trace, trace2)),
            "observation_equal": all(x[1] == y[1] for x, y in zip(trace, trace2)),
            "admissible_equal": all(x[2] == y[2] for x, y in zip(trace, trace2)),
            "branch_reproducible": reproducible,
            "branches_differ": len({fp for _, fp in outs}) > 1,
            "distinct_states_visited": len({t[0] for t in trace})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/environment.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)["env"]
    print(json.dumps(system_report(), indent=2))
    counts = {s: len(list_games(cfg["data_root"], s)) for s in ("train", "valid_seen", "valid_unseen")}
    print("games per split (after ALFWorld filters):", counts)
    if min(counts.values()) == 0:
        # never report a vacuous PASS: zero games means the data path is wrong
        print(f"REPLAY CHECK: FAIL (no games found under {cfg['data_root']})")
        sys.exit(1)
    games = list_games(cfg["data_root"], "valid_seen")
    env = AlfWorldEnv(max_steps=cfg["max_steps"])
    all_ok = True
    for i, g in enumerate(games[: args.episodes]):
        r = replay_check(env, g.game_file, n_steps=30, seed=i)
        ok = all(r[k] for k in ("fingerprint_equal", "observation_equal", "admissible_equal", "branch_reproducible"))
        all_ok &= ok
        print(g.task_id, json.dumps(r), "OK" if ok else "FAIL")
    env.close()
    print("REPLAY CHECK:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
