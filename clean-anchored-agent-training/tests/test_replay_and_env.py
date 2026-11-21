"""Replay semantics, checkpoints, fault injection, error separation, splits."""
import collections
import json

from src.environment.bfcl_env import (ALL_INJECTED, EVAL_ERRORS, TRAIN_ERRORS, V2_ERRORS, Episode, obs_kind)
from src.environment.tasks import leakage_report
from src.replay.replay import bind_args, new_instances, parse_call, run_calls, state_hash


def _task_with_positional(tasks):
    import ast
    for t in tasks:
        for c in t["prefix_gt"] + t["turn_gt"]:
            if ast.parse(c, mode="eval").body.args:
                return t, c
    raise AssertionError("no positional ground-truth call found")


def test_positional_args_are_not_dropped(tasks):
    t, c = _task_with_positional(tasks)
    name, kw, pos = parse_call(c)
    assert pos, c
    inst = new_instances(t["entry"])
    run_calls(inst, t["prefix_gt"])
    bound = bind_args(inst, name, kw, pos)
    assert len(bound) == len(kw) + len(pos)
    assert all(v in bound.values() for v in pos)


def test_checkpoint_restore_is_exact(tasks):
    t = next(x for x in tasks if x["split"] == "train" and len(x["turn_gt"]) >= 2)
    ep = Episode(t, "S1", 0, error_pool=TRAIN_ERRORS)
    snap = ep.snapshot()
    h = state_hash(ep.inst)
    run_calls(ep.inst, t["turn_gt"])
    assert state_hash(ep.inst) != h                       # the episode really changed state
    assert state_hash(snap.inst) == h                     # the checkpoint did not
    a, b = snap.snapshot(), snap.snapshot()
    run_calls(a.inst, t["turn_gt"]); run_calls(b.inst, t["turn_gt"])
    assert state_hash(a.inst) == state_hash(b.inst)       # deterministic replay from the same checkpoint


def test_fault_semantics(tasks):
    t = next(x for x in tasks if x["split"] == "train")
    f = t["fault_fn"]
    n, kw, pos = parse_call(next(c for c in t["turn_gt"] if parse_call(c)[0] == f))
    for sc in ("S1", "S2", "S3"):
        ep = Episode(t, sc, 0, error_pool=TRAIN_ERRORS)
        run_calls(ep.inst, [])                              # no-op
        for c in t["turn_gt"]:
            m, k2, p2 = parse_call(c)
            if m == f:
                break
            ep.call(m, bind_args(ep.inst, m, k2, p2))
        o = ep.call(f, bind_args(ep.inst, f, kw, pos))
        assert o in TRAIN_ERRORS and obs_kind(o) == "injected"
    ep = Episode(t, "S2", 0, error_pool=TRAIN_ERRORS)
    assert obs_kind(ep.call(ep.alt, bind_args(ep.inst, f, kw, pos))) != "injected"   # alternative works in S2


def test_error_pools_disjoint_and_real_errors_untouched():
    assert not (set(V2_ERRORS) & set(TRAIN_ERRORS)) and not (set(TRAIN_ERRORS) & set(EVAL_ERRORS)) and not (set(V2_ERRORS) & set(EVAL_ERRORS))
    assert obs_kind('{"error": "No user is currently logged in."}') == "native"
    assert all(obs_kind(e) == "injected" for e in ALL_INJECTED)


def test_splits_family_grouped_and_no_text_leakage(tasks):
    rep = leakage_report(tasks)
    assert rep["base_tasks_in_multiple_splits"] == 0
    assert rep["user_text_shared_across_splits"] == 0
    for t in tasks:
        if "VehicleControlAPI" in t["entry"]["involved_classes"]:
            assert t["split"] == "heldout"
        if t["split"] == "heldout":
            assert "VehicleControlAPI" in t["entry"]["involved_classes"]


def test_heldout_gt_sequences_not_in_train(tasks):
    tr = collections.Counter(json.dumps(t["turn_gt"]) for t in tasks if t["split"] == "train")
    assert not any(json.dumps(t["turn_gt"]) in tr for t in tasks if t["split"] == "heldout")
