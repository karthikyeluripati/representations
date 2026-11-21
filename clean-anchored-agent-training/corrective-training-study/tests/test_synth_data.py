"""CPU checks of the synthetic-API diversity data (docs/method_synthetic.md)."""
import json
import os
import sys

import pytest

STUDY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(STUDY, "src"))
pytest.importorskip("bfcl_eval")
VERSIONS = [v for v in ("synth_v1", "synth_v2_64") if os.path.exists(os.path.join(STUDY, "data", v, "syn_iar.jsonl"))]
pytestmark = pytest.mark.skipif(not VERSIONS, reason="no synthetic data built")


@pytest.fixture(params=VERSIONS)
def load(request):
    d = os.path.join(STUDY, "data", request.param)
    return lambda n: [json.loads(l) for l in open(os.path.join(d, n), encoding="utf-8")]


def test_iar_convention_and_decisiveness(load):
    from src.environment.bfcl_env import TRAIN_ERRORS
    for r in load("syn_iar.jsonl"):
        f = r["msgs"][2]["tool_calls"][0]["function"]
        err = r["msgs"][3]["content"]
        if r["kind"] == "syn_check":
            assert err in TRAIN_ERRORS[1:] and r["chosen"]["name"].lower().startswith(("list", "get", "view", "show"))
        elif r["kind"] == "syn_post":
            obs = r["msgs"][-1]["content"]
            applied = "final_text" in r["chosen"]
            assert (f["arguments"][next(iter(f["arguments"]))] in obs) == applied          # decisive by construction
            if not applied:
                assert r["chosen"]["name"] == f["name"] and r["chosen"]["arguments"] == f["arguments"]
        else:                                                                              # rejection / idempotent
            assert r["chosen"]["name"] == f["name"]
            if r["kind"] == "syn_reject_retry":
                assert err == TRAIN_ERRORS[0]


def test_old_convention_differs_only_on_create_timeouts(load):
    from src.environment.bfcl_env import TRAIN_ERRORS
    old = load("syn_old.jsonl")
    assert len(old) == len(load("syn_iar.jsonl"))
    for r in old:
        assert r["chosen"]["name"] == r["msgs"][2]["tool_calls"][0]["function"]["name"]      # always a direct retry
    assert any(r["kind"] == "syn_old_retry" and r["msgs"][3]["content"] in TRAIN_ERRORS[1:] for r in old)


def test_no_eval_wordings_and_no_bfcl_function_names(load):
    from src.environment.bfcl_env import AC_EVAL_UNSEEN_MSGS, AC_MESSAGES, load_bfcl
    bfcl_names = {d["name"] for docs in load_bfcl()[2].values() for d in docs}
    for n in ("syn_iar.jsonl", "syn_old.jsonl"):
        for r in load(n):
            blob = json.dumps(r["msgs"], ensure_ascii=False)
            assert all(AC_MESSAGES[m] not in blob for m in AC_EVAL_UNSEEN_MSGS)
            assert not ({t["function"]["name"] for t in r["tools"]} - {"escalate_to_user"}) & bfcl_names


def test_64_class_set_matches_32_class_set_in_size_and_composition():
    import collections
    d1, d2 = (os.path.join(STUDY, "data", v) for v in ("synth_v1", "synth_v2_64"))
    if not all(os.path.exists(os.path.join(d, "syn_iar.jsonl")) for d in (d1, d2)):
        pytest.skip("both sets needed")
    a = [json.loads(l) for l in open(os.path.join(d1, "syn_iar.jsonl"), encoding="utf-8")]
    b = [json.loads(l) for l in open(os.path.join(d2, "syn_iar.jsonl"), encoding="utf-8")]
    assert collections.Counter(r["kind"] for r in a) == collections.Counter(r["kind"] for r in b)
    assert len({r["task_id"] for r in a}) == 32 and len({r["task_id"] for r in b}) == 64
    fn = lambda rs: {t["function"]["name"] for r in rs for t in r["tools"]}
    assert len(fn(b) - {"escalate_to_user"}) == 4 * 64                     # all 256 function names distinct
