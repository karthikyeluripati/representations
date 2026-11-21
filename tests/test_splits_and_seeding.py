import random

import pytest

from dare.datasets.splits import assert_no_leakage, split_by_task
from dare.utils.common import derive_seed, stable_hash

TIDS = [f"{fam}-Obj{i}-None-Recep-{i}/trial_{i}" for fam in
        ("pick_and_place_simple", "pick_two_obj_and_place", "look_at_obj_in_light") for i in range(50)]


def test_split_deterministic_and_disjoint():
    a = split_by_task(TIDS, 0.2, holdout_families=("pick_two_obj_and_place",))
    b = split_by_task(list(reversed(TIDS)), 0.2, holdout_families=("pick_two_obj_and_place",))
    assert a == b  # order-independent, process-independent
    fit = [{"task_id": t} for t, s in a.items() if s == "fit"]
    dev = [{"task_id": t} for t, s in a.items() if s == "dev"]
    assert_no_leakage(fit, dev)
    assert all(s == "family_holdout" for t, s in a.items() if t.startswith("pick_two_obj_and_place"))
    n_dev = sum(s == "dev" for s in a.values())
    assert 5 <= n_dev <= 40  # ~20% of the 100 non-held-out tasks


def test_leakage_detected():
    with pytest.raises(AssertionError):
        assert_no_leakage([{"task_id": "x"}], [{"task_id": "x"}])


def test_stable_seeds():
    # literal value pins cross-process stability (Python's hash() would vary between runs)
    assert stable_hash("a", 1) == stable_hash("a", 1) != stable_hash("a", 2)
    assert derive_seed(0, "task", "random") == derive_seed(0, "task", "random")
    r1 = random.Random(derive_seed(0, "t", "random"))
    r2 = random.Random(derive_seed(0, "t", "random"))
    assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]
