"""Action-selection policies over a matched candidate set.

All policies select from the *same* candidate set (ALFWorld admissible commands,
optionally truncated to `max_candidates`) so comparisons are budget-matched.
LM-based policies (frozen, cosine-novelty, elliptical-novelty) all pay exactly
one prefix pass + one batched candidate pass per step; the novelty bonus itself
is O(K*H^2) numpy and its wall time is logged in `select_s`.

Naming honesty:
  * ``cosine_novelty``  — generic max-min cosine-distance novelty. NOT RepExp.
  * ``elliptical_novelty`` — a per-step, episode-level *adaptation* inspired by
    RepExp's elliptical bonus h^T Sigma^{-1} h (Tuyls et al., 2025). RepExp
    applies the bonus to select among complete sampled responses (inference) or
    as a group-level RL reward; here it is applied to candidate *actions* within
    an episode, with Sigma accumulated over embeddings of previously selected
    actions. It is not a faithful reproduction.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from dare.agents.lm import LMScorer, build_prompt_messages, static_user_prefix


@dataclass
class Decision:
    action: str
    index: int
    scores: list[float] | None = None          # policy score per candidate (as used for selection)
    lm_logp: list[float] | None = None         # raw LM scores if computed
    bonus: list[float] | None = None
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    lm_s: float = 0.0
    select_s: float = 0.0
    context_emb: np.ndarray | None = field(default=None, repr=False)
    action_emb: np.ndarray | None = field(default=None, repr=False)
    prompt_tokens_computed: int = 0


class Policy:
    name = "base"
    uses_lm = False

    def begin_episode(self, rng: random.Random, task_family: str | None = None):
        self.rng = rng
        self.task_family = task_family

    def act(self, intro: str, history: list[tuple[str, str]], observation: str, candidates: list[str]) -> Decision:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def act(self, intro, history, observation, candidates):
        t0 = time.perf_counter()
        i = self.rng.randrange(len(candidates))
        return Decision(candidates[i], i, select_s=time.perf_counter() - t0)


class LMPolicy(Policy):
    """Frozen LM policy: scores candidates by log-likelihood, greedy (temperature=0) or softmax sampling."""
    name = "frozen_lm"
    uses_lm = True

    def __init__(self, scorer: LMScorer, score_type: str = "mean", temperature: float = 0.0, max_history: int = 8,
                 demos: dict[str, str] | None = None):
        self.scorer, self.score_type, self.temperature, self.max_history = scorer, score_type, temperature, max_history
        self.demos = demos or {}

    def _score(self, intro, history, observation, candidates):
        demo = self.demos.get(getattr(self, "task_family", None))
        msgs = build_prompt_messages(intro, history, observation, candidates, self.max_history, demo)
        r = self.scorer.score(msgs, candidates, static_user_prefix(intro, demo))
        lp = r.logp_mean if self.score_type == "mean" else r.logp_sum
        return r, lp

    def _choose(self, values: np.ndarray) -> int:
        if self.temperature <= 0:
            return int(np.argmax(values))
        z = (values - values.max()) / self.temperature
        p = np.exp(z) / np.exp(z).sum()
        return int(self.rng.choices(range(len(values)), weights=p.tolist())[0])

    def act(self, intro, history, observation, candidates):
        r, lp = self._score(intro, history, observation, candidates)
        t0 = time.perf_counter()
        i = self._choose(lp)
        return Decision(candidates[i], i, lp.tolist(), lp.tolist(), None, r.prompt_tokens, r.candidate_tokens,
                        r.wall_s, time.perf_counter() - t0, r.context_emb, r.action_emb[i], r.prompt_tokens_computed)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class CosineNoveltyPolicy(LMPolicy):
    """score = lm_logp + lam * min_j (1 - cos(h_a, h_j)) over previously selected action embeddings."""
    name = "cosine_novelty"

    def __init__(self, scorer, lam: float = 1.0, **kw):
        super().__init__(scorer, **kw)
        self.lam = lam

    def begin_episode(self, rng, task_family=None):
        super().begin_episode(rng, task_family)
        self.memory: list[np.ndarray] = []

    def act(self, intro, history, observation, candidates):
        r, lp = self._score(intro, history, observation, candidates)
        t0 = time.perf_counter()
        E = _unit(r.action_emb)
        if self.memory:
            M = np.stack(self.memory)
            bonus = (1.0 - E @ M.T).min(1)
        else:
            bonus = np.ones(len(candidates))
        s = lp + self.lam * bonus
        i = self._choose(s)
        self.memory.append(E[i])
        return Decision(candidates[i], i, s.tolist(), lp.tolist(), bonus.tolist(), r.prompt_tokens,
                        r.candidate_tokens, r.wall_s, time.perf_counter() - t0, r.context_emb, r.action_emb[i], r.prompt_tokens_computed)


class EllipticalNoveltyPolicy(LMPolicy):
    """RepExp-inspired adaptation: score = lm_logp + beta * h^T Sigma^{-1} h,
    Sigma = reg*I + sum_j h_j h_j^T over selected (unit-normalised) action embeddings
    in the current episode; Sigma^{-1} maintained by Sherman–Morrison updates."""
    name = "elliptical_novelty"

    def __init__(self, scorer, beta: float = 1.0, reg: float = 1.0, **kw):
        super().__init__(scorer, **kw)
        self.beta, self.reg = beta, reg

    def begin_episode(self, rng, task_family=None):
        super().begin_episode(rng, task_family)
        self.inv = None

    def act(self, intro, history, observation, candidates):
        r, lp = self._score(intro, history, observation, candidates)
        t0 = time.perf_counter()
        # No per-step mean-centring: Sigma accumulates across steps, so all vectors must share one frame.
        E = _unit(r.action_emb)
        if self.inv is None:
            self.inv = np.eye(E.shape[1]) / self.reg
        bonus = ((E @ self.inv) * E).sum(1)  # BLAS matmul; einsum over H=2048 was ~2x slower per step
        s = lp + self.beta * bonus
        i = self._choose(s)
        h = E[i][:, None]
        Ah = self.inv @ h
        self.inv = self.inv - (Ah @ Ah.T) / (1.0 + (h.T @ Ah).item())
        return Decision(candidates[i], i, s.tolist(), lp.tolist(), bonus.tolist(), r.prompt_tokens,
                        r.candidate_tokens, r.wall_s, time.perf_counter() - t0, r.context_emb, r.action_emb[i], r.prompt_tokens_computed)


def make_policy(name: str, scorer: LMScorer | None, cfg: dict, demos: dict[str, str] | None = None) -> Policy:
    pc = cfg.get("policy", {})
    common = dict(score_type=pc.get("score_type", "mean"), temperature=pc.get("temperature", 0.0),
                  max_history=pc.get("max_history", 8), demos=demos)
    if name == "random":
        return RandomPolicy()
    if scorer is None:
        raise ValueError(f"policy {name} needs an LM scorer")
    if name == "frozen_lm":
        return LMPolicy(scorer, **common)
    if name == "cosine_novelty":
        return CosineNoveltyPolicy(scorer, lam=pc.get("novelty_lambda", 1.0), **common)
    if name == "elliptical_novelty":
        return EllipticalNoveltyPolicy(scorer, beta=pc.get("novelty_beta", 1.0), reg=pc.get("novelty_reg", 1.0), **common)
    raise ValueError(name)
