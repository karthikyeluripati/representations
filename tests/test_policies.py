"""Policy logic with a fake LM scorer (no model weights needed)."""
import random

import numpy as np

from dare.agents.lm import ScoreResult
from dare.agents.policies import CosineNoveltyPolicy, EllipticalNoveltyPolicy, LMPolicy, RandomPolicy


class FakeScorer:
    def __init__(self, embs, logp):
        self.embs, self.logp = np.asarray(embs, float), np.asarray(logp, float)

    def score(self, msgs, cands, static_prefix=None):
        K = len(cands)
        return ScoreResult(self.logp[:K], self.logp[:K], np.ones(K, int), np.zeros(3), self.embs[:K], 10, K, 0.0)


CANDS = ["a", "b", "c"]
EMBS = [[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]]


def test_frozen_greedy_and_random_determinism():
    p = LMPolicy(FakeScorer(EMBS, [-1.0, -0.5, -2.0]))
    p.begin_episode(random.Random(0))
    assert p.act("", [], "", CANDS).action == "b"
    r1, r2 = RandomPolicy(), RandomPolicy()
    r1.begin_episode(random.Random(5)); r2.begin_episode(random.Random(5))
    assert [r1.act("", [], "", CANDS).index for _ in range(10)] == [r2.act("", [], "", CANDS).index for _ in range(10)]


def test_novelty_moves_away_from_chosen_direction():
    for P in (CosineNoveltyPolicy(FakeScorer(EMBS, [0, 0, 0]), lam=1.0),
              EllipticalNoveltyPolicy(FakeScorer(EMBS, [0, 0, 0]), beta=1.0)):
        P.begin_episode(random.Random(0))
        first = P.act("", [], "", CANDS).action
        second = P.act("", [], "", CANDS).action
        assert first in ("a", "b") and second == "c", P.name  # after a/b, the orthogonal c is most novel


def test_elliptical_inverse_matches_direct():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(6, 4))
    P = EllipticalNoveltyPolicy(FakeScorer(E, np.zeros(6)), beta=1.0, reg=0.5)
    P.begin_episode(random.Random(0))
    chosen = []
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    for _ in range(3):
        d = P.act("", [], "", [str(i) for i in range(6)])
        chosen.append(U[d.index])
    S = 0.5 * np.eye(4) + sum(np.outer(h, h) for h in chosen)
    assert np.allclose(P.inv, np.linalg.inv(S), atol=1e-8)
