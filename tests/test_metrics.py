import numpy as np
import pytest
from scipy.stats import spearmanr

from dare.evaluation.metrics import (auroc, brier, cluster_bootstrap_ci, expected_calibration_error,
                                     pairwise_ranking_accuracy, spearman, wilson_interval)

sk = pytest.importorskip("sklearn.metrics")


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 200).astype(float)
    y = rng.integers(0, 2, 200)
    assert auroc(s, y) == pytest.approx(sk.roc_auc_score(y, s))
    assert np.isnan(auroc(s, np.ones(200)))


def test_pairwise_ranking():
    # group A perfectly ordered, group B inverted; cross-group pairs ignored
    s = [1, 2, 3, 3, 2, 1]
    t = [0, 1, 2, 0, 1, 2]
    g = ["A"] * 3 + ["B"] * 3
    assert pairwise_ranking_accuracy(s, t, g) == pytest.approx(0.5)
    assert pairwise_ranking_accuracy([1, 1], [0, 1], ["A", "A"]) == 0.5  # tie
    assert np.isnan(pairwise_ranking_accuracy([1, 2], [1, 1], ["A", "A"]))


def test_spearman_brier_ece():
    x = np.arange(10.0)
    y = x ** 3
    assert spearman(x, y) == pytest.approx(1.0)
    assert spearman(x, -y) == pytest.approx(spearmanr(x, -y).statistic)
    assert brier([1, 0], [1, 0]) == 0 and brier([0.5], [1]) == 0.25
    assert expected_calibration_error([0.2] * 10, [0] * 8 + [1] * 2) == pytest.approx(0.0)
    assert expected_calibration_error([0.9] * 4, [0] * 4) == pytest.approx(0.9)


def test_cluster_bootstrap_resamples_clusters():
    vals = np.array([0.0] * 5 + [1.0] * 5)
    clusters = np.array([0] * 5 + [1] * 5)
    point, lo, hi = cluster_bootstrap_ci(lambda idx: vals[idx].mean(), clusters, n_boot=500)
    assert point == 0.5
    # with 2 clusters each resample mean is one of {0, .5, 1}
    assert lo == 0.0 and hi == 1.0


def test_wilson():
    lo, hi = wilson_interval(0, 24)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.1 < hi < 0.2
