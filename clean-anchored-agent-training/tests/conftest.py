import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
PAIRS = os.path.join(ROOT, "data", "cacrt_pairs_v1")


@pytest.fixture(scope="session")
def tasks():
    return json.load(open(os.path.join(ROOT, "data", "tasks_v1.json")))


@pytest.fixture(scope="session")
def by_id(tasks):
    return {t["id"]: t for t in tasks}


def jl(name):
    return [json.loads(l) for l in open(os.path.join(PAIRS, name))]


@pytest.fixture(scope="session")
def train_states():
    return jl("train_states.jsonl")
