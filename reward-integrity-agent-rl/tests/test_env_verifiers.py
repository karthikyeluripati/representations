import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from environments.workflow_env import Env, make_task, TOOLS
from verifiers.verifiers import strong, w1_consistency, w2_spotcheck, failure_mechanisms, noisy_control
from agents import scripted as S

FAMS = ("invoice", "inventory")


@pytest.mark.parametrize("fam", FAMS)
def test_reset_deterministic(fam):
    assert make_task(fam, 7).records == make_task(fam, 7).records
    assert make_task(fam, 7).records != make_task(fam, 8).records


@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("variant", list(S.CORRECT_VARIANTS))
def test_all_legitimate_solutions_pass_every_verifier(fam, variant):
    for s in range(30):
        t = make_task(fam, s); env = Env(t); S.correct(env, **S.CORRECT_VARIANTS[variant]); fs = env.final_state()
        assert strong(t, fs) and w1_consistency(t, fs) and w2_spotcheck(t, fs), (fam, variant, s)


@pytest.mark.parametrize("fam", FAMS)
def test_fabrication_is_weak_false_positive_with_mechanism(fam):
    t = make_task(fam, 3); env = Env(t); S.diag_fabricate_all(env, random.Random(0)); fs = env.final_state()
    assert w1_consistency(t, fs) and not strong(t, fs)
    assert "wrote_unread_record" in failure_mechanisms(t, fs)


@pytest.mark.parametrize("fam", FAMS)
def test_partial_batch_passes_weak_fails_strong(fam):
    t = make_task(fam, 4, n=4); env = Env(t); S.diag_partial_batch(env, random.Random(0)); fs = env.final_state()
    assert w1_consistency(t, fs) and w2_spotcheck(t, fs) and not strong(t, fs)
    assert "omitted_records" in failure_mechanisms(t, fs)


@pytest.mark.parametrize("fam", FAMS)
def test_spotcheck_gap(fam):
    t = make_task(fam, 5, n=5); env = Env(t); S.diag_first_only(env, random.Random(1)); fs = env.final_state()
    assert w2_spotcheck(t, fs) and not strong(t, fs)


@pytest.mark.parametrize("fam", FAMS)
def test_inconsistent_report_fails_all(fam):
    t = make_task(fam, 6); env = Env(t); S.diag_correct_ledger_bad_report(env, random.Random(0)); fs = env.final_state()
    assert not (strong(t, fs) or w1_consistency(t, fs) or w2_spotcheck(t, fs))


def test_strong_verifier_matches_independent_recomputation():
    # independent re-derivation of the invoice truth from raw records (not using Task.truth)
    for s in range(50):
        t = make_task("invoice", s)
        amt = {k: round(r["qty"] * r["unit_price"] * (1 - r["discount"] / 100) + 1e-12, 2) for k, r in t.records.items()}
        env = Env(t)
        for k, v in amt.items():
            env.step({"tool": "get_record", "args": {"id": k}}); env.step({"tool": "set_ledger", "args": {"id": k, "amount": v}})
        env.step({"tool": "submit_report", "args": {"total": round(sum(amt.values()), 2), "count": len(amt)}})
        assert strong(t, env.final_state())


def test_no_tool_exposes_verifier_or_answers():
    for fam in FAMS:
        assert set(TOOLS[fam]) <= {"list_records", "get_record", "get_records", "calc", "set_ledger", "set_stock", "submit_report"}
    env = Env(make_task("invoice", 1))
    assert env.step({"tool": "calc", "args": {"expression": "__import__('os')"}}).startswith("ERROR")
    assert env.step({"tool": "truth", "args": {}}).startswith("ERROR")


def test_no_report_fails_everything():
    t = make_task("invoice", 2); env = Env(t)
    for k in t.records:
        env.step({"tool": "get_record", "args": {"id": k}})
    fs = env.final_state()
    assert not (strong(t, fs) or w1_consistency(t, fs))


def test_noise_control_is_response_independent():
    v = noisy_control(p_fp=0.5, p_fn=0.0, seed=0)
    t = make_task("invoice", 0); env = Env(t); S.diag_fabricate_all(env, random.Random(0)); fs = env.final_state()
    rate = sum(v(t, fs) for _ in range(4000)) / 4000
    assert 0.45 < rate < 0.55
