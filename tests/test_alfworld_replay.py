"""Integration tests against real ALFWorld games (skipped where TextWorld/data are unavailable)."""
import os

import pytest

pytest.importorskip("textworld")
pytest.importorskip("alfworld")

from dare.agents.policies import RandomPolicy  # noqa: E402
from dare.datasets.branching import StateMismatch, estimate_branch, restore  # noqa: E402
from dare.environments.alfworld_env import AlfWorldEnv, list_games, select_games  # noqa: E402
from dare.evaluation.runner import run_episode  # noqa: E402
from dare.evaluation.verify import verify_episodes  # noqa: E402
from dare.utils.logging import RunLogger, load_run  # noqa: E402

ROOT = "~/.cache/alfworld/json_2.1.1"
if not os.path.isdir(os.path.expanduser(ROOT)):
    pytest.skip("ALFWorld data not downloaded", allow_module_level=True)


@pytest.fixture(scope="module")
def games():
    return select_games(list_games(ROOT, "valid_seen"), 3)


@pytest.fixture()
def env():
    e = AlfWorldEnv(max_steps=15)
    yield e
    e.close()


def test_random_episode_reproducible(env, games):
    s1 = run_episode(env, RandomPolicy(), games[0], seed=3, run_name="t")
    s2 = run_episode(env, RandomPolicy(), games[0], seed=3, run_name="t")
    s3 = run_episode(env, RandomPolicy(), games[0], seed=4, run_name="t")
    assert s1["actions"] == s2["actions"] and s1["final_state_fp"] == s2["final_state_fp"]
    assert s1["actions"] != s3["actions"]


def test_replay_restores_state_and_detects_mismatch(env, games):
    s = run_episode(env, RandomPolicy(), games[1], seed=0, run_name="t")
    k = len(s["actions"]) // 2
    ref = env.replay(games[1].game_file, s["actions"][:k])
    fresh = AlfWorldEnv(max_steps=15)
    st = restore(fresh, games[1], s["actions"][:k], ref.fingerprint, ref.observation, ref.admissible)
    assert st.fingerprint == ref.fingerprint
    with pytest.raises(StateMismatch):
        restore(fresh, games[1], s["actions"][:k], "0" * 16)
    fresh.close()


def test_branch_estimates_random_rollouts(env, games):
    st = env.reset(games[2].game_file)
    out = estimate_branch(env, RandomPolicy(), games[2], [], st.fingerprint, st.admissible[0], n_rollouts=3,
                          base_seed=0)
    assert out["rollout_count"] == 3 and 0 <= out["branch_success_rate"] <= 1
    assert 1 <= out["estimated_remaining_actions"] <= 15
    again = estimate_branch(env, RandomPolicy(), games[2], [], st.fingerprint, st.admissible[0], 3, 0)
    assert again["rollout_outcomes"] == out["rollout_outcomes"]


def test_logged_run_verifies_and_tampering_is_caught(tmp_path, env, games):
    lg = RunLogger(tmp_path, meta={"config_hash": "x"})
    for g in games[:2]:
        run_episode(env, RandomPolicy(), g, seed=0, run_name="t", logger=lg)
    eps, trs = load_run(tmp_path)
    rep = verify_episodes(eps, trs, max_steps=15)
    assert rep["ok"] and rep["n_steps"] == sum(e["n_steps"] for e in eps) > 0
    eps[0] = {**eps[0], "success": not eps[0]["success"]}
    trs[1] = {**trs[1], "next_observation": "tampered"}
    rep = verify_episodes(eps, trs, max_steps=15)
    kinds = {m[2] for m in rep["mismatches"]}
    assert not rep["ok"] and "final outcome" in kinds and "next_observation" in kinds


def test_world_fingerprint_ignores_history_bookkeeping(env, games):
    s0 = env.reset(games[1].game_file)
    s1 = env.step("look")
    assert s1.fingerprint != s0.fingerprint  # strict hash sees checked(...) bookkeeping
    assert s1.world_fp == s0.world_fp        # world state unchanged -> counts as wasted
    locs = [a for a in s0.admissible if a.startswith("go to")]
    a = env.step(locs[0])
    env.step(locs[1])
    back = env.step(locs[0])
    assert back.world_fp == a.world_fp and back.fingerprint != a.fingerprint
    takes = [c for c in back.admissible if c.startswith("take ")]
    if takes:
        assert env.step(takes[0]).world_fp != back.world_fp  # real state change is not waste
