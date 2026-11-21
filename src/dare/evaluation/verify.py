"""Verify that logged trajectories correspond to actual environment behaviour.

For each completed episode, replay its logged actions in a *fresh* environment
and check, step by step, that the observation, symbolic-state fingerprint,
admissible set, and won/done flags match what was logged; and that the logged
episode outcome equals the environment's goal check at the end.
"""
from __future__ import annotations

from collections import defaultdict

from dare.environments.alfworld_env import AlfWorldEnv


def verify_episodes(episodes: list[dict], transitions: list[dict], max_steps: int) -> dict:
    by_ep = defaultdict(list)
    for t in transitions:
        by_ep[t["episode_id"]].append(t)
    env = AlfWorldEnv(max_steps=max_steps)
    report = {"n_episodes": 0, "n_steps": 0, "mismatches": []}
    try:
        for e in episodes:
            trs = sorted(by_ep[e["episode_id"]], key=lambda t: t["step"])
            if [t["selected_action"] for t in trs] != e["actions"]:
                report["mismatches"].append((e["episode_id"], -1, "actions in summary != transitions"))
            st = env.reset(e["game_file"])
            for t in trs:
                checks = {"state_fp": st.fingerprint == t["state_fp"], "observation": st.observation == t["observation"],
                          "candidates": set(t["candidates"]) <= set(st.admissible)}
                st = env.step(t["selected_action"])
                checks.update({"next_state_fp": st.fingerprint == t["next_state_fp"],
                               "next_observation": st.observation == t["next_observation"],
                               "won": st.won == t["won"], "done": st.done == t["done"]})
                if "next_world_fp" in t:
                    checks["next_world_fp"] = st.world_fp == t["next_world_fp"]
                for k, ok in checks.items():
                    if not ok:
                        report["mismatches"].append((e["episode_id"], t["step"], k))
                report["n_steps"] += 1
            if st.won != e["success"] or st.fingerprint != e["final_state_fp"]:
                report["mismatches"].append((e["episode_id"], len(trs), "final outcome"))
            report["n_episodes"] += 1
    finally:
        env.close()
    report["ok"] = not report["mismatches"]
    return report
