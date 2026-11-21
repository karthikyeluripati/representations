"""CPU checks of the outcome-twin (OTV) training data: twin structure, execution verification, decisiveness, and that
no evaluation task, class or wording leaks into training. Needs bfcl_eval importable; data/otv_v1 must be built."""
import json
import os
import sys

import pytest

STUDY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(STUDY, "src")); sys.path.insert(0, os.path.join(STUDY, "scripts"))
pytest.importorskip("bfcl_eval")
DATA = os.path.join(STUDY, "data", "otv_v1", "pairs_abstract.jsonl")
pytestmark = pytest.mark.skipif(not os.path.exists(DATA), reason="data/otv_v1 not built")


@pytest.fixture(scope="module")
def ctx():
    from src.environment.bfcl_env import AC_MESSAGES, AC_EVAL_UNSEEN_MSGS, AC_TRAIN_MSGS
    pairs = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    tasks = {t["id"]: t for t in json.load(open(os.path.join(PROJ, "data", "tasks_v1.json"), encoding="utf-8"))}
    feas = {r["task_id"]: r for r in json.load(open(os.path.join(PROJ, "results", "feasibility_ambiguous_commit.json"), encoding="utf-8"))}
    return pairs, tasks, feas, AC_MESSAGES, AC_EVAL_UNSEEN_MSGS, AC_TRAIN_MSGS


def test_split_has_no_leakage(ctx):
    pairs, tasks, feas, AC, UNSEEN, TRAIN = ctx
    for p in pairs:
        t, r = tasks[p["task_id"]], feas[p["task_id"]]
        assert t["split"] == "train" and t["fault_class"] in ("TravelAPI", "TradingBot") and r["decisive"]
        blob = json.dumps(p["msgs"], ensure_ascii=False)
        for m in UNSEEN:
            assert AC[m] not in blob, f"evaluation wording {m} leaked into training"
    assert not set(TRAIN) & set(UNSEEN)


def test_twins_share_history_and_diverge_after_check(ctx):
    pairs = ctx[0]
    by = {}
    for p in pairs:
        by.setdefault((p["task_id"], p.get("ac_msg"), p["type"]), []).append(p)
    n = 0
    for (tid, m, ty), ps in by.items():
        if ty == "otv_a":
            assert len({json.dumps(p["msgs"]) for p in ps}) == 1          # one shared error state
            assert {p["neg"] for p in ps} == {"retry", "skip", "escalate"}
        if ty == "otv_b":
            a = next(p for p in ps if p["twin"] == "ACa"); l = next(p for p in ps if p["twin"] == "ACl")
            assert a["msgs"][:-1] == l["msgs"][:-1]                         # identical up to the check's observation
            assert a["msgs"][-1] != l["msgs"][-1]                           # the observation differs
            assert a["chosen"] == l["rejected"] and a["rejected"] == l["chosen"]   # opposite decisions
            n += 1
    assert n > 0


def test_post_check_observation_is_decisive(ctx):
    """In ACa the check shows some argument value of f that is absent in ACl's check."""
    pairs = ctx[0]
    for p in pairs:
        if p["type"] == "otv_b" and p["twin"] == "ACa":
            f = p["rejected"]                                  # re-issuing f is wrong in ACa
            twin = next(q for q in pairs if q["type"] == "otv_b" and q["twin"] == "ACl" and q["state"] == p["state"].replace("|ACa|", "|ACl|"))
            oa, ol = p["msgs"][-1]["content"], twin["msgs"][-1]["content"]
            vals = [v if isinstance(v, str) else json.dumps(v) for v in f["arguments"].values()]
            assert any(len(v) >= 3 and v in oa and v not in ol for v in vals), p["state"]


def test_pairs_reverify_by_execution(ctx):
    """Rebuild a sample of tasks from scratch and require identical pairs (builder is deterministic and verified)."""
    pairs, tasks, feas, *_ = ctx
    from build_otv_pairs import verifier_for  # noqa
    from src.environment.bfcl_env import load_bfcl
    from verification.outcome_twins import build_task_pairs
    docs = load_bfcl()[2]
    ids = sorted({p["task_id"] for p in pairs})[:4]
    for tid in ids:
        ps, rep = build_task_pairs(tasks[tid], verifier_for(tasks[tid], feas[tid], docs), "explicit")
        o = rep["outcomes"]
        assert o["verify|ACa"] and o["verify|ACl"] and not o["retry|ACa"] and not o["skip|ACl"] and not o["escalate|ACa"]
        strip = lambda xs: [{k: v for k, v in x.items() if k != "evidence_value"} for x in xs]   # field added in v2
        assert [x for x in pairs if x["task_id"] == tid and x.get("ac_msg") == "explicit"] == strip(ps)


@pytest.mark.parametrize("ver,classes", [("otv_v2", ("TravelAPI", "TradingBot", "TwitterAPI", "TicketAPI")),
                                         ("otv_v2T", ("MessageAPI", "TradingBot", "TwitterAPI", "TicketAPI"))])
def test_evidence_claims_are_true_and_folds_do_not_leak(ver, classes):
    d = os.path.join(STUDY, "data", ver)
    if not os.path.exists(os.path.join(d, "pairs_evidence.jsonl")):
        pytest.skip(f"{ver} not built")
    from src.environment.bfcl_env import AC_EVAL_UNSEEN_MSGS, AC_MESSAGES
    tasks = {t["id"]: t for t in json.load(open(os.path.join(PROJ, "data", "tasks_v1.json"), encoding="utf-8"))}
    ev = [json.loads(l) for l in open(os.path.join(d, "pairs_evidence.jsonl"), encoding="utf-8")]
    pl = [json.loads(l) for l in open(os.path.join(d, "pairs_plain.jsonl"), encoding="utf-8")]
    assert [p["state"] for p in ev] == [p["state"] for p in pl] and len(ev) > 0
    for p in ev:
        t = tasks[p["task_id"]]
        assert t["split"] == "train" and t["fault_class"] in classes          # held-out class never trained on
        assert all(AC_MESSAGES[m] not in json.dumps(p["msgs"], ensure_ascii=False) for m in AC_EVAL_UNSEEN_MSGS)
        if p["type"] != "otv_b":
            continue
        obs, val = p["msgs"][-1]["content"], p["evidence_value"]
        claim = (p["chosen"].get("content") or p["chosen"].get("final_text"))
        shows = val in obs
        assert shows == (p["twin"] == "ACa")                                   # decisive: value visible iff applied
        assert ("the result shows" in claim) == shows and ("does not show" in claim) == (not shows)
        bad = (p["rejected"].get("content") or p["rejected"].get("final_text"))
        assert ("the result shows" in bad) != shows                            # rejected side makes the false claim


def test_iar_sets_differ_only_in_write_timeout_handling():
    d = os.path.join(STUDY, "data", "iar_v1")
    if not os.path.exists(os.path.join(d, "iar.jsonl")):
        pytest.skip("iar_v1 not built")
    from src.environment.bfcl_env import AC_EVAL_UNSEEN_MSGS, AC_MESSAGES, TRAIN_ERRORS
    tasks = {t["id"]: t for t in json.load(open(os.path.join(PROJ, "data", "tasks_v1.json"), encoding="utf-8"))}
    iar = [json.loads(l) for l in open(os.path.join(d, "iar.jsonl"), encoding="utf-8")]
    old = [json.loads(l) for l in open(os.path.join(d, "aac_matched.jsonl"), encoding="utf-8")]
    assert len(iar) == len(old)
    for r in iar + old:
        t = tasks[r["task_id"]]
        assert t["split"] == "train" and t["fault_class"] != "TravelAPI"          # held-out class never trained on
        blob = json.dumps(r["msgs"], ensure_ascii=False)
        assert all(AC_MESSAGES[m] not in blob for m in AC_EVAL_UNSEEN_MSGS)        # eval wordings never seen
    # IAR answers a timeout-type failure on a state-changing call with a CHECK (never a direct retry) ...
    for r in iar:
        last = r["msgs"][-1]["content"]
        if r["kind"] == "ac_twin_timeout" and last in TRAIN_ERRORS[1:]:
            f = r["msgs"][-2]["tool_calls"][0]["function"]["name"]
            assert r["chosen"].get("name") != f
        if r["kind"] in ("rejection_retry", "idem_retry"):
            assert r["chosen"]["name"] == r["msgs"][-2]["tool_calls"][0]["function"]["name"]
    # ... while the baseline answers the same failures with a direct retry (the old convention)
    wr = [r for r in old if r["kind"] == "write_retry" and r["msgs"][-1]["content"] in TRAIN_ERRORS[1:]]
    assert wr and all(r["chosen"]["name"] == r["msgs"][-2]["tool_calls"][0]["function"]["name"] for r in wr)
