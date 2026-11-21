"""Ranking, calibration and uncertainty metrics (numpy/scipy only)."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy.stats import spearmanr


def pairwise_ranking_accuracy(scores, targets, groups) -> float:
    """Fraction of within-group pairs with different targets that `scores` order correctly.
    Ties in scores count as 0.5. Pairs with equal targets are skipped. Returns nan if no pairs."""
    scores, targets, groups = map(np.asarray, (scores, targets, groups))
    correct, total = 0.0, 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        s, t = scores[idx], targets[idx]
        ds = s[:, None] - s[None, :]
        dt = t[:, None] - t[None, :]
        m = dt > 0  # ordered pairs where i should beat j
        total += int(m.sum())
        correct += float((ds[m] > 0).sum() + 0.5 * (ds[m] == 0).sum())
    return correct / total if total else float("nan")


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def auroc(scores, labels) -> float:
    """Mann–Whitney AUROC with tie correction. nan if one class is absent."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def brier(probs, outcomes) -> float:
    p, o = np.asarray(probs, float), np.asarray(outcomes, float)
    return float(np.mean((p - o) ** 2))


def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    """Equal-width-bin ECE. `outcomes` may be fractional (e.g. branch success rates)."""
    p, o = np.asarray(probs, float), np.asarray(outcomes, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - o[m].mean())
    return float(ece)


def cluster_bootstrap_ci(stat_fn: Callable[[np.ndarray], float], clusters, n_boot: int = 2000,
                         alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile CI resampling whole clusters (e.g. tasks) with replacement.

    `stat_fn` receives an index array into the original rows and returns a scalar.
    Returns (point_estimate, lo, hi); nan resamples are dropped.
    """
    clusters = np.asarray(clusters)
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = list(groups)
    idx_lists = [np.array(groups[k]) for k in keys]
    rng = np.random.default_rng(seed)
    point = stat_fn(np.arange(len(clusters)))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        v = stat_fn(np.concatenate([idx_lists[k] for k in pick]))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)
