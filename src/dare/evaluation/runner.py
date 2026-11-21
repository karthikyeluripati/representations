"""Run one logged episode of a policy in ALFWorld."""
from __future__ import annotations

import random
import time

from dare.agents.policies import Decision, Policy
from dare.environments.alfworld_env import AlfWorldEnv, GameInfo
from dare.utils.common import derive_seed


def episode_id_for(run_name: str, policy: str, task_id: str, seed: int) -> str:
    return f"{run_name}|{policy}|{task_id}|s{seed}"


def run_episode(env: AlfWorldEnv, policy: Policy, game: GameInfo, seed: int, run_name: str,
                logger=None, max_candidates: int | None = None, token_budget: int | None = None,
                model_id: str | None = None, cfg_hash: str | None = None, prefix: list[str] | None = None) -> dict:
    """Play one episode. `prefix` optionally forces the first actions (used for branching).

    Wasted action (operational definition): an action after which the *world*
    state (PDDL facts minus history-bookkeeping predicates, see
    BOOKKEEPING_PREDICATES) equals one already visited in this episode: a no-op
    such as look/examine/inventory/help/failed command, or a return to a previous
    world state (e.g. walking back to a location without changing anything).
    """
    ep_id = episode_id_for(run_name, policy.name, game.task_id, seed)
    rng = random.Random(derive_seed(seed, game.task_id, policy.name))
    policy.begin_episode(rng, game.task_family)
    t_ep = time.perf_counter()
    st = env.reset(game.game_file)
    intro = st.observation
    history: list[tuple[str, str]] = []
    visited = {st.world_fp}
    actions, n_wasted, tot_prompt, tot_cand = [], 0, 0, 0
    prefix = list(prefix or [])
    while not st.done:
        cands = st.admissible if max_candidates is None else st.admissible[:max_candidates]
        forced = bool(prefix)
        if forced:
            a = prefix.pop(0)
            if a not in st.admissible:
                raise ValueError(f"forced prefix action {a!r} not admissible at step {len(actions)}")
            dec = Decision(a, cands.index(a) if a in cands else -1)
        else:
            if token_budget is not None and tot_prompt + tot_cand >= token_budget:
                break
            dec = policy.act(intro, history, st.observation, cands)
        prev = st
        st = env.step(dec.action)
        wasted = st.world_fp in visited
        visited.add(st.world_fp)
        n_wasted += int(wasted)
        tot_prompt += dec.prompt_tokens
        tot_cand += dec.candidate_tokens
        if logger is not None:
            logger.log_transition({
                "episode_id": ep_id, "step": len(actions), "task_id": game.task_id, "split": game.split,
                "task_family": game.task_family, "game_file": game.game_file, "seed": seed,
                "policy": policy.name, "model_id": model_id, "config_hash": cfg_hash,
                "objective": env.objective, "observation": prev.observation,
                "history_ref": {"episode_id": ep_id, "n_prev_steps": len(actions)},
                "candidates": cands, "selected_action": dec.action, "selected_index": dec.index,
                "policy_scores": dec.scores, "lm_logp": dec.lm_logp, "novelty_bonus": dec.bonus,
                "next_observation": st.observation, "reward": st.reward, "won": st.won, "done": st.done,
                "state_fp": prev.fingerprint, "next_state_fp": st.fingerprint,
                "world_fp": prev.world_fp, "next_world_fp": st.world_fp, "wasted": wasted,
                "forced_prefix": forced,
                "tokens": {"prompt": dec.prompt_tokens, "prompt_computed": dec.prompt_tokens_computed,
                           "candidates": dec.candidate_tokens,
                           "generated": 0, "total": dec.prompt_tokens + dec.candidate_tokens},
                "time_s": {"lm": dec.lm_s, "select": dec.select_s, "env": st.env_time_s},
            })
        history.append((dec.action, st.observation))
        actions.append(dec.action)
    summary = {
        "episode_id": ep_id, "run_name": run_name, "policy": policy.name, "task_id": game.task_id,
        "split": game.split, "task_family": game.task_family, "game_file": game.game_file, "seed": seed,
        "model_id": model_id, "config_hash": cfg_hash, "objective": env.objective,
        "success": bool(st.won), "n_steps": len(actions), "n_wasted": n_wasted, "actions": actions,
        "final_state_fp": st.fingerprint, "budget_exhausted": (not st.done),
        "tokens_total": tot_prompt + tot_cand, "tokens_prompt": tot_prompt, "tokens_candidates": tot_cand,
        "wall_s": time.perf_counter() - t_ep,
    }
    if logger is not None:
        logger.end_episode(summary)
    return summary
