import json, os, sys, collections
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import pytest
from environments.bfcl_faults import Episode, reference_policy, load_bfcl, ERRORS, parse_call, public_state, new_instances, run_calls
from agents.rollout import initial_messages, openai_tools, SYSTEM_BASE
from training.paired_data import build, assistant_indices

TASKS = json.load(open(os.path.join(ROOT, "data", "tasks.json")))
_, _, DOCS = load_bfcl()
SAMPLE = TASKS[::7]


@pytest.mark.parametrize("sc", ["S0", "S1", "S2", "S3"])
def test_reference_policy_correct(sc):
    for t in SAMPLE:
        for seed in (0, 3):
            ep = Episode(t, sc, seed); reference_policy(ep)
            assert ep.grade()["correct"], (t["id"], sc, seed)


def test_s3_impossible_class_cannot_change():
    """In S3 every method of the fault class fails, so its state stays at the episode start; ground truth needs a change."""
    for t in SAMPLE:
        ep = Episode(t, "S3", 0)
        start = public_state(ep.inst[t["fault_class"]])
        for c in t["turn_gt"]:
            n, a = parse_call(c); ep.call(n, a); ep.call(n + "_backup", a)
        assert public_state(ep.inst[t["fault_class"]]) == start
        gt = new_instances(t["entry"]); run_calls(gt, t["prefix_gt"] + t["turn_gt"])
        assert public_state(gt[t["fault_class"]]) != start
        assert not ep.grade()["state_ok"]


def test_prompts_and_tools_identical_across_scenarios():
    for t in SAMPLE:
        m = [json.dumps(initial_messages(t, SYSTEM_BASE), default=str) for _ in ("S0", "S1", "S2", "S3")]
        assert len(set(m)) == 1
        txt = m[0] + json.dumps(openai_tools(t, DOCS))
        for leak in ("S1", "S2", "S3", "scenario", "impossible", "recoverab"):
            assert leak not in txt


def test_first_fault_observation_identical_across_failing_scenarios():
    for t in SAMPLE:
        n, a = parse_call([c for c in t["turn_gt"] if parse_call(c)[0] == t["fault_fn"]][0])
        obs = set()
        for sc in ("S1", "S2", "S3"):
            ep = Episode(t, sc, 0)
            for c in t["turn_gt"]:          # replay up to the fault call
                cn, ca = parse_call(c)
                if cn == t["fault_fn"]:
                    break
                ep.call(cn, ca)
            if sc == "S3" and any(o in ERRORS for _, _, o in ep.calls):
                continue                    # S3 may already have failed on an earlier method of the same class
            obs.add(ep.call(n, a))
        assert len(obs) == 1 and obs.pop() in ERRORS


def test_escalation_on_solvable_is_false_abstention():
    t = SAMPLE[0]
    for sc in ("S0", "S1", "S2"):
        ep = Episode(t, sc, 0); ep.call("escalate_to_user", {"reason": "x"})
        g = ep.grade(); assert g["escalated"] and not g["correct"]
    ep = Episode(t, "S3", 0); ep.call("escalate_to_user", {"reason": "x"}); assert ep.grade()["correct"]


def test_splits_keep_base_tasks_together_and_heldout_family():
    by = collections.defaultdict(set)
    for t in TASKS:
        by[t["base_id"]].add(t["split"])
    assert all(len(v) == 1 for v in by.values())
    assert all((t["split"] == "test_heldout_family") == (t["fault_class"] == "VehicleControlAPI") for t in TASKS)


def test_pairs_diverge_only_after_distinguishing_evidence():
    tr, pairs = build([t for t in TASKS if t["split"] == "train"][:15], DOCS)
    assert pairs
    for p in pairs:
        a, b = p["solvable"], p["impossible"]
        ia, ib = p["i_solvable"], p["i_impossible"]
        # identical assistant actions before the divergence point
        pa = [i for i in assistant_indices(a) if i < ia]; pb = [i for i in assistant_indices(b) if i < ib]
        assert [json.dumps(a["msgs"][i].get("tool_calls"), sort_keys=True) for i in pa] == [json.dumps(b["msgs"][i].get("tool_calls"), sort_keys=True) for i in pb]
        # the observation right before the divergence differs (success vs error)
        assert a["msgs"][ia - 1]["role"] == "tool" and b["msgs"][ib - 1]["role"] == "tool"
        assert a["msgs"][ia - 1]["content"] != b["msgs"][ib - 1]["content"]
        assert b["msgs"][ib - 1]["content"] in ERRORS
