"""CPU validation for the matched CF-SFT pilot (docs/cf_sft_pilot_protocol.md): T2 generator, oracle validity,
splits/leakage, and the three training sets (data/cf_sft_v1)."""
import collections
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from verification import semantic_cf as T1  # noqa: E402
from verification import semantic_cf_t2 as T  # noqa: E402
from verification import synthetic_apis as SA  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "cf_sft_v1")
TEST, DEV, TRAIN = T.specs("test"), T.specs("dev"), T.specs("train")


def _play(s, plan):
    f = T.FamilyT2(s["family"]); w = f.names[s["name"]]; r = f.read_names[s["verifier"]]
    base = f.w_args(s["effect"]); nokey = {k: v for k, v in base.items() if k != "request_key"}
    ep = T.T2Episode(s)
    def W(a=base):
        return ep.call(w, a)
    if plan == "retry":
        W()
    elif plan == "retry_nokey":
        W(nokey)
    elif plan == "escalate":
        ep.call(T.ESC, {"reason": "x"})
    elif plan == "check_branch":
        o = ep.call(r, {"account_id": f.acct}); (None if T._shows_applied(o, f.amount) else W())
    elif plan == "reconcile":
        o = json.loads(W()); act = o.get("todays_active_transactions") or []
        if len(act) > 1:
            ep.call(f.reverse, {"transaction_id": act[0]["transaction_id"]})
    ep.finish(); return ep.grade()["correct"]


# ---------------- splits and leakage
def test_vocabulary_disjoint_across_templates_and_splits():
    t1 = {d for d, *_ in T1.FAMILIES} | {n for _, n, _ in T1.FAMILIES}; syn = {d for d, *_ in SA.DOMAINS} | {n for _, n, _ in SA.DOMAINS}
    assert len(set(T.DOMAINS_T2)) == 60 and not set(T.DOMAINS_T2) & (t1 | syn)
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not set(T.VERBS[a]) & set(T.VERBS[b])
        assert not {t for e in T.CONTRACT[a].values() for t in e} & {t for e in T.CONTRACT[b].values() for t in e}
        assert not set(T.ERRORS[a].values()) & set(T.ERRORS[b].values()) - {T.ERRORS["test"]["reject"]}
    assert T.ERRORS["test"]["timeout_a"] == T1.ERRORS["implicit"]          # the realistic wording is held out from T2 train


def test_heldout_combos_only_outside_training():
    assert not any(s["heldout_combo"] for s in T.specs("train", exclude_heldout_combos=True))
    assert {(s["effect"], s["verifier"], s["compensation"]) for s in TEST if s["heldout_combo"]} == T.HELDOUT_COMBOS


# ---------------- oracle validity: safe retry vs necessary check vs reconcile vs unresolvable
def test_escalation_wrong_whenever_feasible_and_right_only_when_unresolvable():
    for s in TEST:
        if s["error"] == "none":
            continue
        feas = T.oracle(s)[0]
        assert feas is not (s["error"] in T.TIMEOUTS and s["effect"] == "nonidem" and s["verifier"] != "decisive" and s["compensation"] == "absent")
        assert _play(s, "escalate") is (not feas)


def test_safe_retries_without_verifier():
    for s in TEST:
        if s["error"] in T.TIMEOUTS and s["verifier"] != "decisive" and s["effect"] in ("idem", "keyed"):
            assert _play(s, "retry")
        if s["error"] in T.TIMEOUTS and s["effect"] == "keyed":
            assert _play(s, "retry_nokey") is (s["outcome"] == "lost")        # dropping the key duplicates the applied twin


def test_check_necessary_for_nonidem_with_decisive_verifier():
    for s in TEST:
        if s["error"] in T.TIMEOUTS and s["effect"] == "nonidem" and s["verifier"] == "decisive":
            assert _play(s, "check_branch") and (_play(s, "retry") is (s["outcome"] == "lost" or False) or s["compensation"] == "present")


def test_reconcile_solves_unobservable_nonidem_when_compensation_exists():
    for s in TEST:
        if s["error"] in T.TIMEOUTS and s["effect"] == "nonidem" and s["verifier"] != "decisive" and s["compensation"] == "present":
            assert _play(s, "reconcile") and _play(s, "retry") is (s["outcome"] == "lost")


def test_twins_identical_until_a_write_when_not_decisive():
    for s in TEST:
        if s["error"] in T.TIMEOUTS and s["verifier"] != "decisive" and s["outcome"] == "applied":
            a, l = T.T2Episode(s), T.T2Episode(dict(s, outcome="lost")); f = a.f
            assert a.messages("x") == l.messages("x") and a.tools() == l.tools()
            args = {"account_id": f.acct} if s["verifier"] != "none" else {}
            assert a.call(f.read_names[s["verifier"]], args) == l.call(f.read_names[s["verifier"]], args)


def test_separation_table():
    m = {n: T.metrics(T.run_policy(P(), TEST)) for n, P in T.POLICIES_T2.items()}
    c = m["contract_policy"]
    for k in ("decision_accuracy", "A_effect_sensitivity", "infeasible_safe_escalation", "reconcile_twin_correct", "keyed_retry_correct",
              "outcome_twin_correct", "feasible_end_state", "hc_decision_accuracy", "fault_free_success", "post_check_interpretation"):
        assert c[k]["mean"] == 1.0, k
    assert c["unnecessary_check_rate"]["mean"] == 0 and c["excess_calls"]["mean"] == 0
    for n in ("always_retry", "always_check"):
        assert m[n]["A_effect_sensitivity"]["mean"] == 0 and m[n]["infeasible_safe_escalation"]["mean"] == 0
        assert m[n]["reconcile_twin_correct"]["mean"] == 0
    assert m["always_check"]["unnecessary_check_rate"]["mean"] > 0.5


# ---------------- training sets
@pytest.fixture(scope="module")
def arms():
    if not os.path.exists(os.path.join(DATA, "MANIFEST.json")):
        pytest.skip("run scripts/build_cf_sft_data.py first")
    return {n: [json.loads(l) for l in open(os.path.join(DATA, f"{n}.jsonl"), encoding="utf-8")] for n in ("CF", "CTRL", "ABL")}


def test_sizes_and_shared_base(arms):
    man = json.load(open(os.path.join(DATA, "MANIFEST.json")))
    assert all(len(v) == 1349 for v in arms.values())
    base = [json.dumps(r, sort_keys=True) for r in arms["CF"][:1029]]
    assert all([json.dumps(r, sort_keys=True) for r in v[:1029]] == base for v in arms.values())
    for n, v in arms.items():
        import hashlib
        assert hashlib.sha256(open(os.path.join(DATA, f"{n}.jsonl"), "rb").read()).hexdigest() == man["arms"][n]["sha256"]


def test_no_dev_test_or_t1_strings_in_training(arms):
    held = [T.DOMAINS_T2[i] for i in list(T.SPLITS["dev"]) + list(T.SPLITS["test"])] + [v for s in ("dev", "test") for v in T.VERBS[s]]
    held += [t for s in ("dev", "test") for e in T.CONTRACT[s].values() for t in e]
    held += [T.ERRORS[s][k] for s in ("dev", "test") for k in ("timeout_a", "timeout_b")]
    held += list(T1.CONTRACT["test"]["idem"]) + list(T1.CONTRACT["test"]["nonidem"])
    t1_ids = {x for i in range(len(T1.FAMILIES)) for x in list(T1.Family(i).names.values()) + list(T1.Family(i).read_names.values())}
    heldout_domains = [T.DOMAINS_T2[k] for k in list(T.SPLITS["dev"]) + list(T.SPLITS["test"])]
    for n, v in arms.items():
        txt = "\n".join(json.dumps(r, ensure_ascii=False) for r in v[1029:])
        bad = [h for h in held if h in txt]
        assert not bad, (n, bad[:5])
        ids = {t["function"]["name"] for r in v[1029:] for t in r["tools"]}      # T1 / dev / test identifiers never appear
        assert not ids & t1_ids and not any(d in i for d in heldout_domains for i in ids)


def test_added_records_valid(arms):
    for n, v in arms.items():
        add = v[1029:]
        assert len(add) == 320 and len({r["state"] for r in add}) == 320
        assert not any(r["spec"]["heldout_combo"] or r["spec"]["split"] != "train" for r in add)
        for r in add:
            names = {t["function"]["name"] for t in r["tools"]}
            assert "final_text" in r["chosen"] or r["chosen"]["name"] in names
        if n == "CTRL":                                   # never a cell where the CAW rule is unsafe
            assert not any(r["spec"]["effect"] == "nonidem" and r["spec"]["verifier"] != "decisive" and r["spec"]["error"] in T.TIMEOUTS for r in add)
        if n == "ABL":
            txt = json.dumps([r["tools"] for r in add])
            assert not any(t in txt for s in T.CONTRACT.values() for e in s.values() for t in e)
    cf = arms["CF"][1029:]
    roles = collections.Counter(r["role"] for r in cf)
    assert roles["escalate"] == 45 and roles["reconcile_reverse"] == 25 and roles["retry_idem"] == 30
    assert sum(r["disagrees_with_caw"] for r in cf) >= 100                   # real counterfactual signal vs the CAW rule
    assert [r["chosen"] for r in arms["ABL"][1029:]] == [r["chosen"] for r in cf]


def test_builder_deterministic(tmp_path, arms):
    import build_cf_sft_data as B
    a = B.balanced(B.records(T.T2ContractPolicy(), T.specs("train", exclude_heldout_combos=True)[:600], "cf")[0], {"ff_call": 5}, 0)
    b = B.balanced(B.records(T.T2ContractPolicy(), T.specs("train", exclude_heldout_combos=True)[:600], "cf")[0], {"ff_call": 5}, 0)
    assert [x["state"] for x in a] == [x["state"] for x in b]


def test_prompts_frozen_and_distinct():
    assert T.PROMPTS["P1"] == T1.PROMPT_HINT and T.system_for("P0") == T1.SYSTEM
    assert len({T.system_for(p) for p in T.PROMPTS}) == 4
    held = [T.DOMAINS_T2[i] for i in T.SPLITS["test"]] + T.VERBS["test"] + [t for e in T.CONTRACT["test"].values() for t in e]
    assert not any(h in T.PROMPTS["P3"] for h in held)
