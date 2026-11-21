"""Counterfactual replay semantics: valid states, same-state branching, no splicing, verified outcomes, data isolation."""
import json

from conftest import jl

from src.counterfactuals.construct import Counter, make_episode, run_branch
from src.counterfactuals.policies import OutageRule, Repairer
from src.environment.bfcl_env import EVAL_ERRORS, obs_kind
from src.replay.replay import state_hash


def test_states_rebuild_to_stored_hash(by_id, train_states):
    c = Counter()
    for s in train_states[:80]:
        ep = make_episode(by_id[s["task_id"]], s["forced"], c)
        assert state_hash(ep.inst) == s["state_hash"]
        assert obs_kind(ep.calls[-1][2]) == "native"               # every corrective state ends in a REAL error
        assert ep.calls[-1][2] == s["error_obs"]


def test_retained_pairs_are_outcome_verified(by_id, train_states):
    c = Counter()
    kept = [s for s in train_states if s["retained_pairs"]][:60]
    assert kept
    for s in kept:
        t = by_id[s["task_id"]]
        snap = make_episode(t, s["forced"], c)
        h = state_hash(snap.inst)
        pos = (s["pos_action"]["name"], s["pos_action"]["arguments"])
        ok, _ = run_branch(snap, pos, s["repair_plan"][1:], OutageRule(), c)
        assert ok
        for nm in s["retained_pairs"]:
            a = s["neg_actions"][nm]
            bad, _ = run_branch(snap, (a["name"], a["arguments"]), s["repair_plan"][1:], OutageRule(), c)
            assert not bad
        assert state_hash(snap.inst) == h                           # branching never mutates the shared checkpoint
        assert s["naive_fails"]


def test_repairer_isolates_escalation_as_outcome_critical(by_id, train_states):
    """Under a competent continuation, retrying a real error is wasteful but recoverable; escalation is fatal."""
    c = Counter()
    for s in [x for x in train_states if x["retained_pairs"] and x["kind"] == "RA"][:20]:
        snap = make_episode(by_id[s["task_id"]], s["forced"], c)
        r = s["neg_actions"]["retry"]
        ok, _ = run_branch(snap, (r["name"], r["arguments"]), s["repair_plan"][1:], Repairer(s["repair_plan"]), c)
        assert ok
        e = s["neg_actions"]["escalate"]
        ok, _ = run_branch(snap, (e["name"], e["arguments"]), s["repair_plan"][1:], Repairer(s["repair_plan"]), c)
        assert not ok


def test_twin_anchors_differ_only_in_the_observation(train_states):
    tw = jl("train_anchors_twin.jsonl")
    states = {s["key"]: s for s in train_states}
    for a in tw[:100]:
        s = states[a["of"]]
        last_obs = a["messages"][-1]["content"]
        if a["kind"] == "T-out":
            assert obs_kind(last_obs) == "injected" and last_obs not in EVAL_ERRORS
            assert a["action"]["name"] == s["pivot_call"].split("(")[0]          # correct = identical retry
        else:
            assert obs_kind(last_obs) == "ok"
        # same user request and history as the corrective state
        assert [m["content"] for m in a["messages"] if m["role"] == "user"] == [m["content"] for m in s["messages"] if m["role"] == "user"]


def test_pairs_share_the_exact_prefix(train_states):
    from src.baselines.arms import pair_records
    for p in pair_records(train_states)[:200]:
        assert p["chosen"] != p["rejected"]
        assert p["msgs"][-1]["role"] == "tool" and obs_kind(p["msgs"][-1]["content"]) == "native"


def test_training_data_never_touches_eval_tasks(tasks):
    train_ids = {t["id"] for t in tasks if t["split"] == "train"}
    for f in ("train_states.jsonl", "train_anchors_twin.jsonl", "train_anchors_generic.jsonl"):
        for r in jl(f):
            assert r["task_id"] in train_ids
            blob = json.dumps(r)
            assert not any(e in blob for e in EVAL_ERRORS) and "_replica" not in blob
