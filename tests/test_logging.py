import json

import numpy as np
import pytest

from dare.utils.logging import RunLogger, load_run


def _ep(logger, ep_id, n=3):
    for s in range(n):
        logger.log_transition({"episode_id": ep_id, "step": s, "emb": np.zeros(2), "x": np.float32(1.5)})
    logger.end_episode({"episode_id": ep_id, "actions": [f"a{s}" for s in range(n)]})


def test_roundtrip_and_numpy(tmp_path):
    lg = RunLogger(tmp_path, meta={"config_hash": "h"})
    _ep(lg, "e1")
    eps, trs = load_run(tmp_path)
    assert [e["episode_id"] for e in eps] == ["e1"]
    assert [t["step"] for t in trs] == [0, 1, 2]
    assert trs[0]["emb"] == [0.0, 0.0] and trs[0]["x"] == 1.5
    # summary and transitions share the attempt id (regression: summary was once written without it)
    assert eps[0]["attempt"] == trs[0]["attempt"] == lg.attempt


def test_incomplete_episode_dropped_and_rerun_not_duplicated(tmp_path):
    lg = RunLogger(tmp_path, meta={"config_hash": "h"})
    _ep(lg, "e1")
    # simulate a crash: transitions of e2 hit disk, its summary never does
    with open(tmp_path / "transitions.jsonl", "a") as f:
        f.write(json.dumps({"episode_id": "e2", "step": 0, "attempt": "crashed"}) + "\n")
        f.write('{"episode_id": "e2", "st')  # torn line
    eps, trs = load_run(tmp_path)
    assert {t["episode_id"] for t in trs} == {"e1"}
    lg2 = RunLogger(tmp_path, meta={"config_hash": "h"})  # resume
    assert lg2.completed_episode_ids() == {"e1"}
    _ep(lg2, "e2", n=2)
    eps, trs = load_run(tmp_path)
    assert sorted(t["step"] for t in trs if t["episode_id"] == "e2") == [0, 1]


def test_resume_refuses_changed_config(tmp_path):
    RunLogger(tmp_path, meta={"config_hash": "a"})
    with pytest.raises(RuntimeError):
        RunLogger(tmp_path, meta={"config_hash": "b"})
