"""CPU tests for the semantic-counterfactual diagnostic (docs/semantic_counterfactual_protocol.md)."""
import collections
import itertools
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from verification import semantic_cf as S  # noqa: E402
from verification import synthetic_apis as SA  # noqa: E402

TEST = S.specs("test"); TRAIN = S.specs("train")


def _ep_after(spec, actions):
    ep = S.CFEpisode(spec); outs = [ep.call(n, a) for n, a in actions]; return ep, outs


def _args(spec):
    f = S.Family(spec["family"]); return f, {f.id_key: f.rid}


def test_no_overlap_with_training_vocabulary():
    doms = {d for d, *_ in SA.DOMAINS}; nouns = {n for _, n, _ in SA.DOMAINS}
    assert not doms & {d for d, *_ in S.FAMILIES} and not nouns & {n for _, n, _ in S.FAMILIES}
    syn_verbs = set(SA.CREATE_VERBS + SA.SET_VERBS + SA.REMOVE_VERBS + SA.LIST_VERBS)
    test_verbs = {v for v, _ in S.VERB0["test"]} | {v.split("_")[0] for v, _ in S.ALT["test"]}
    assert not test_verbs & syn_verbs
    train_verbs = {v for v, _ in S.VERB0["train"]} | {v.split("_")[0] for v, _ in S.ALT["train"]}
    assert not test_verbs & train_verbs
    assert {S.Family(s["family"]).split for s in TEST} == {"test"} and {S.Family(s["family"]).split for s in TRAIN} == {"train"}


def test_contract_templates_split():
    train_t = [t for e in S.CONTRACT["train"].values() for t in e]
    for s in TEST[:200]:
        d = S.CFEpisode(s).tools()[0]["function"]["description"]
        assert not any(t.split("{")[0][:25] in d for t in train_t)


def test_names_never_predict_effect():
    c = collections.Counter((s["family"], s["name"], s["effect"]) for s in TEST)
    for (fam, name, eff), n in c.items():
        assert c[(fam, name, "idem" if eff == "nonidem" else "nonidem")] == n
    for i in range(len(S.FAMILIES)):
        f = S.Family(i); assert f.names["n0"] != f.names["n1"]


def test_twins_observationally_identical_without_decisive_verifier():
    for s in TEST:
        if s["error"] not in S.TIMEOUTS or s["verifier"] in S.DECISIVE or s["outcome"] != "applied":
            continue
        lost = dict(s, outcome="lost"); f, a = _args(s); r = f.read_names[s["verifier"]]
        acts = [(r, a), (f.names[s["name"]], a), (r, a)]
        ea, oa = _ep_after(s, acts); el, ol = _ep_after(lost, acts)
        assert oa == ol and ea.tools() == el.tools()
        assert ea.messages("x") == el.messages("x")    # identical histories, including the error string


def test_decisive_verifier_separates_twins():
    for s in TEST:
        if s["error"] in S.TIMEOUTS and s["verifier"] in S.DECISIVE and s["outcome"] == "applied":
            f, a = _args(s); r = f.read_names[s["verifier"]]
            assert _ep_after(s, [(r, a)])[1] != _ep_after(dict(s, outcome="lost"), [(r, a)])[1]


def test_history_lost_observation_is_error_shaped():
    s = next(x for x in TEST if x["verifier"] == "history" and x["outcome"] == "lost" and x["error"] == "implicit")
    f, a = _args(s); assert _ep_after(s, [(f.read_names["history"], a)])[1][0].startswith("Error: no changes recorded")


@pytest.mark.parametrize("eff,out,reissue,correct", [("nonidem", "applied", True, False), ("nonidem", "lost", True, True),
                                                     ("idem", "applied", True, True), ("idem", "lost", False, False)])
def test_grader_decisive(eff, out, reissue, correct):
    s = next(x for x in TEST if x["effect"] == eff and x["outcome"] == out and x["verifier"] == "listing" and x["error"] == "implicit")
    f, a = _args(s); ep, _ = _ep_after(s, [(f.names[s["name"]], a)] if reissue else []); ep.finish()
    g = ep.grade(); assert g["correct"] is correct and g["duplicate"] is (eff == "nonidem" and out == "applied" and reissue)


def test_grader_infeasible_requires_escalation_without_reissue():
    s = next(x for x in TEST if x["effect"] == "nonidem" and x["verifier"] == "none" and x["error"] == "implicit" and x["outcome"] == "lost")
    f, a = _args(s)
    assert _ep_after(s, [(S.ESC, {"reason": "x"})])[0].grade()["correct"]
    assert not _ep_after(s, [(f.names[s["name"]], a)])[0].grade()["correct"]           # retry completes by luck: not credited
    assert not _ep_after(s, [(f.names[s["name"]], a), (S.ESC, {"reason": "x"})])[0].grade()["correct"]
    ep = S.CFEpisode(s); ep.finish("Done."); assert not ep.grade()["correct"]               # silent finish is not safe


def test_separation_table():
    m = {n: S.metrics(S.run_policy(P(), TEST, S.SYSTEM)) for n, P in S.POLICIES.items()}
    keys = ("decision_accuracy", "A_effect_sensitivity", "B_name_both_correct", "twin_joint_listing", "twin_joint_history",
            "infeasible_safe_escalation", "feasible_end_state", "fault_free_success")
    assert all(m["contract_policy"][k]["mean"] == 1.0 for k in keys) and m["contract_policy"]["unnecessary_check_rate"]["mean"] == 0
    for n in ("always_retry", "escalate_on_timeout", "check_first_esc_on_error", "check_first_reissue_on_error"):
        assert m[n]["A_effect_sensitivity"]["mean"] == 0.0
    assert m["check_first_esc_on_error"]["twin_joint_history"]["mean"] == 0.0 and m["check_first_esc_on_error"]["twin_joint_listing"]["mean"] == 1.0
    assert m["name_heuristic"]["B_name_consistency"]["mean"] < 0.9
    assert m["always_retry"]["infeasible_safe_escalation"]["mean"] == 0.0


def test_deterministic():
    assert json.dumps(S.specs("test"), sort_keys=True) == json.dumps(TEST, sort_keys=True)
    assert [S.Family(i).names for i in range(40)] == [S.Family(i).names for i in range(40)]


def test_cluster_ci_paired():
    est, (lo, hi), n = S.cluster_ci({0: 1.0, 1: 0.0, 2: 1.0}, {0: 0.0, 1: 0.0, 2: 0.0})
    assert n == 3 and abs(est - 2 / 3) < 1e-3 and lo <= est <= hi


def test_system_prompt_matches_training_prompt():
    pytest.importorskip("bfcl_eval")
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    from src.counterfactuals.construct import SYSTEM
    assert SYSTEM == S.SYSTEM


# ---- validity of the oracle: safe retries vs necessary verification vs genuinely unresolvable (edit log 1) ----
def _cell(**kw):
    return [s for s in TEST if all(s[k] == v for k, v in kw.items())]


def _play(s, plan):
    f, a = _args(s); r = f.read_names[s["verifier"]]; w = f.names[s["name"]]
    acts = {"retry": [(w, a)], "escalate": [(S.ESC, {"reason": "x"})], "check_then_retry": [(r, a), (w, a)], "check_then_finish": [(r, a)]}[plan]
    ep, _ = _ep_after(s, acts); ep.finish(); return ep.grade()["correct"]


def test_escalation_is_wrong_whenever_completion_is_possible():
    feas = [s for s in TEST if S.oracle(s)[0] and s["error"] != "none"]
    assert {s["verifier"] for s in feas} == set(S.VERIFIERS)          # incl. no-verifier cells (idempotent W)
    assert not any(_play(s, "escalate") for s in feas)


def test_safe_retry_without_any_verifier():
    for s in _cell(effect="idem", error="implicit") :
        if s["verifier"] in ("none", "snapshot"):
            assert _play(s, "retry") and not _play(s, "escalate")


def test_verification_necessary_and_sufficient_for_nonidem_with_decisive_verifier():
    for s in _cell(effect="nonidem", error="implicit"):
        if s["verifier"] in S.DECISIVE:
            right = "check_then_finish" if s["outcome"] == "applied" else "check_then_retry"
            wrong = "check_then_retry" if s["outcome"] == "applied" else "check_then_finish"
            assert _play(s, right) and not _play(s, wrong) and not _play(s, "escalate")
            assert _play(s, "retry") is (s["outcome"] == "lost")       # blind retry only lucky in the lost twin


def test_unresolvable_only_when_nonidem_and_no_decisive_verifier():
    for s in TEST:
        feasible = S.oracle(s)[0]
        assert feasible is not (s["error"] in S.TIMEOUTS and s["effect"] == "nonidem" and s["verifier"] not in S.DECISIVE)
        if not feasible:
            assert _play(s, "escalate") and not _play(s, "retry")


def test_always_check_scored_on_correctness_and_cost():
    m = S.metrics(S.run_policy(S.AlwaysCheck(), TEST, S.SYSTEM))
    c = S.metrics(S.run_policy(S.ContractPolicy(), TEST, S.SYSTEM))
    assert m["twin_joint_listing"]["mean"] == 1.0 and m["twin_joint_history"]["mean"] == 1.0   # verification itself works
    assert m["infeasible_safe_escalation"]["mean"] == 0.0 and m["duplicate_rate"]["mean"] > 0     # but unsafe when unresolvable
    assert m["unnecessary_check_rate"]["mean"] > 0.5 and m["excess_calls"]["mean"] > c["excess_calls"]["mean"] == 0.0
    assert c["initial_decision_correct"]["mean"] == c["post_check_interpretation"]["mean"] == c["outcome_twin_correct"]["mean"] == 1.0
