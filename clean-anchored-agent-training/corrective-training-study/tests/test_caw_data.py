"""CPU checks of the check-all-writes (CAW) data (docs/method_caw.md)."""
import json
import os
import sys

import pytest

STUDY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(STUDY, "src"))
P = os.path.join(STUDY, "data", "caw_v1", "caw.jsonl")
pytestmark = pytest.mark.skipif(not os.path.exists(P), reason="caw_v1 not built")
pytest.importorskip("bfcl_eval")


@pytest.fixture(scope="module")
def rows():
    return [json.loads(l) for l in open(P, encoding="utf-8")]


def test_size_matches_syn64(rows):
    assert len(rows) == 1349 and sum(r["task_id"].startswith("synth:") for r in rows) == 832   # + 84 anchors = 1,433


def test_every_write_timeout_is_checked_reads_and_rejections_retried(rows):
    from src.environment.bfcl_env import TRAIN_ERRORS
    for r in rows:
        if not r["task_id"].startswith("synth:"):
            continue
        f = r["msgs"][2]["tool_calls"][0]["function"]["name"]; err = r["msgs"][3]["content"]
        reads = {t["function"]["name"] for t in r["tools"] if t["function"]["name"].lower().startswith(("list", "get", "view", "show"))}
        if r["kind"] == "caw_check":
            assert f not in reads and err in TRAIN_ERRORS[1:] and r["chosen"]["name"] in reads    # create / set / remove alike
        elif r["kind"] == "caw_post":
            assert ("final_text" in r["chosen"]) or r["chosen"]["name"] == f
        elif r["kind"] == "caw_reject_retry":
            assert err == TRAIN_ERRORS[0] and r["chosen"]["name"] == f
        elif r["kind"] == "caw_read_retry":
            assert f in reads and r["chosen"]["name"] == f
    kinds_checked = {r["state"].split("|")[1] for r in rows if r["kind"] == "caw_check"}
    assert kinds_checked == {"create", "set", "remove"}                       # verb no longer predicts the action


def test_no_idempotent_retry_on_writes_and_no_eval_wordings(rows):
    from src.environment.bfcl_env import AC_EVAL_UNSEEN_MSGS, AC_MESSAGES
    assert not any(r["kind"] in ("idem_retry", "syn_idem_retry") for r in rows)
    for r in rows:
        blob = json.dumps(r["msgs"], ensure_ascii=False)
        assert all(AC_MESSAGES[m] not in blob for m in AC_EVAL_UNSEEN_MSGS)


def test_caw2_persistent_failure_continuations():
    p = os.path.join(STUDY, "data", "caw2_v1", "caw2.jsonl")
    if not os.path.exists(p):
        pytest.skip("caw2_v1 not built")
    from src.environment.bfcl_env import ESCALATE, TRAIN_ERRORS
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert len(rows) == 1349
    pers = [r for r in rows if r["kind"].startswith("persist_")]
    assert len(pers) == 88
    for r in pers:
        tool_obs = [m["content"] for m in r["msgs"] if m["role"] == "tool"]
        assert tool_obs[-1] in TRAIN_ERRORS[1:]                         # the latest attempt failed again
        if r["kind"] == "persist_backup":
            assert r["chosen"]["name"].endswith("_backup") and tool_obs[-2] not in TRAIN_ERRORS   # check succeeded, f failed twice
        else:
            assert r["chosen"]["name"] == ESCALATE and tool_obs[-2] in TRAIN_ERRORS                # the check itself keeps failing
