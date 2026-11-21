"""Counterfactual branch-outcome estimation via deterministic replay.

For a decision state s_t (reached by action prefix a_{<t} in game g) and a
candidate action c:
  1. reset g and replay a_{<t}; assert the symbolic-state fingerprint (hash of
     all PDDL facts), the observation and the admissible set equal the logged
     ones (state equivalence, not just "same actions");
  2. execute c;
  3. continue with the *same frozen rollout policy* until done/truncation;
  4. repeat n_rollouts times with distinct, derived seeds (only meaningful for
     a stochastic rollout policy; greedy policies give identical rollouts).
Outputs per (state, candidate): success rate, mean remaining actions & tokens,
rollout count and a Beta-posterior uncertainty of the success rate.
"""
from __future__ import annotations

import random

import numpy as np

from dare.agents.policies import Policy
from dare.environments.alfworld_env import AlfWorldEnv, GameInfo
from dare.utils.common import derive_seed


class StateMismatch(RuntimeError):
    pass


def restore(env: AlfWorldEnv, game: GameInfo, prefix: list[str], expect_fp: str,
            expect_obs: str | None = None, expect_adm: list[str] | None = None):
    st = env.replay(game.game_file, prefix)
    if st.fingerprint != expect_fp:
        raise StateMismatch(f"fingerprint mismatch after replaying {len(prefix)} actions in {game.task_id}")
    if expect_obs is not None and st.observation != expect_obs:
        raise StateMismatch("observation mismatch after replay")
    if expect_adm is not None and sorted(st.admissible) != sorted(expect_adm):
        raise StateMismatch("admissible-command mismatch after replay")
    return st


def rollout_after(env: AlfWorldEnv, policy: Policy, game: GameInfo, prefix: list[str], expect_fp: str,
                  candidate: str, seed: int, max_candidates: int | None = None) -> dict:
    """Replay prefix, take `candidate`, then follow `policy`. Rebuilds the policy's history from replay."""
    policy.begin_episode(random.Random(seed), game.task_family)
    st = env.reset(game.game_file)
    intro = st.observation
    history = []
    for a in prefix:
        st = env.step(a)
        history.append((a, st.observation))
    if st.fingerprint != expect_fp:
        raise StateMismatch(f"fingerprint mismatch at branch point in {game.task_id}")
    if candidate not in st.admissible:
        raise ValueError(f"candidate {candidate!r} not admissible")
    st = env.step(candidate)
    history.append((candidate, st.observation))
    n_actions, tokens = 1, 0
    while not st.done:
        cands = st.admissible if max_candidates is None else st.admissible[:max_candidates]
        d = policy.act(intro, history, st.observation, cands)
        tokens += d.prompt_tokens + d.candidate_tokens
        st = env.step(d.action)
        history.append((d.action, st.observation))
        n_actions += 1
    return {"success": bool(st.won), "remaining_actions": n_actions, "remaining_tokens": tokens}


def estimate_branch(env, policy, game, prefix, expect_fp, candidate, n_rollouts, base_seed,
                    max_candidates=None) -> dict:
    outs = [rollout_after(env, policy, game, prefix, expect_fp, candidate,
                          derive_seed(base_seed, game.task_id, len(prefix), candidate, r), max_candidates)
            for r in range(n_rollouts)]
    k = sum(o["success"] for o in outs)
    a, b = 1 + k, 1 + n_rollouts - k  # Beta(1,1) prior
    return {
        "candidate_action": candidate,
        "rollout_count": n_rollouts,
        "n_success": k,
        "branch_success_rate": k / n_rollouts,
        "success_posterior_mean": a / (a + b),
        "outcome_uncertainty": float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))),  # Beta posterior std
        "estimated_remaining_actions": float(np.mean([o["remaining_actions"] for o in outs])),
        "estimated_remaining_tokens": float(np.mean([o["remaining_tokens"] for o in outs])),
        "rollout_outcomes": [o["success"] for o in outs],
    }
