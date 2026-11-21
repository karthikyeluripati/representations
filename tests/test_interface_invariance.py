"""Tests for the tool-interface invariance benchmark: execution-verified equivalence, controls and leakage.

Requires a ToolSchemaSpace checkout at the pinned commit: set TSS_PATH (the tests skip without it).
"""
import os, random, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
TSS = os.environ.get("TSS_PATH")
pytestmark = pytest.mark.skipif(not TSS or not os.path.isdir(TSS), reason="TSS_PATH not set")


@pytest.fixture(scope="module")
def data():
    import iface_lib as L
    import construct_paired_transitions as C
    by_dom, queries = L.load_tss(TSS)
    lib = L.tool_library(queries)
    recs = []
    for dom in sorted(by_dom):
        for q in sorted([q for q in queries if q["domain"] == dom], key=lambda q: q["id"])[:6]:
            r, why = C.build_record(q, by_dom, lib, 0)
            assert why is None or not why.startswith("verify"), why
            if r:
                recs.append(r)
    return L, C, by_dom, lib, queries, recs


def test_builds_records_in_every_domain(data):
    L, C, by_dom, lib, q, recs = data
    assert {r["domain"] for r in recs} == set(by_dom)


def test_equivalence_by_execution(data):
    import verify_tool_equivalence as V
    L, C, by_dom, lib, q, recs = data
    for r in recs:
        assert V.verify_record(r, C.build_executors(r, by_dom)) == []


def test_surface_variants_differ_in_text_but_not_state(data):
    L, C, by_dom, lib, q, recs = data
    for r in recs:
        for pk in ("R1", "R1_shuf"):
            base = next(x for x in r["responses"] if x["rid"] == f"{pk}:G:canonical")
            for x in r["responses"]:
                if x["prompt_key"] == pk and x["behavior"] == "G" and x is not base:
                    assert x["final_state_hash"] == base["final_state_hash"]
                    assert (x["text"] == base["text"]) == (x["family"] == "identical")


def test_controls_change_state(data):
    L, C, by_dom, lib, q, recs = data
    for r in recs:
        g = next(x for x in r["responses"] if x["rid"] == "R2_native:G:canonical")["final_state_hash"]
        ctrl = [x for x in r["responses"] if x["behavior"] != "G"]
        assert ctrl and all(x["final_state_hash"] != g for x in ctrl)


def test_order_permutation_changes_sequence_only(data):
    L, C, by_dom, lib, q, recs = data
    n = 0
    for r in recs:
        base = next(x for x in r["responses"] if x["rid"] == "R1:G:canonical")
        for x in r["responses"]:
            if x["rid"] == "R1:G:O":
                n += 1
                assert x["seq_hash"] != base["seq_hash"] and x["final_state_hash"] == base["final_state_hash"]
    assert n > 0


def test_mutated_execution_is_detected(data):
    """Negative test: a wrong argument value inside an alias call must fail verification."""
    import copy, json
    import verify_tool_equivalence as V
    L, C, by_dom, lib, q, recs = data
    r = copy.deepcopy(next(r for r in recs if any(x["family"] == "AB" for x in r["responses"])))
    x = next(x for x in r["responses"] if x["family"] == "AB")
    calls, _ = L.parse(x["text"])
    n, a = calls[0]
    k = next(iter(a))
    a[k] = "__corrupted__" if not isinstance(a[k], (int, float)) else a[k] + 999
    x["text"] = L.render([(n, a)] + calls[1:], L.PREAMBLES[0])
    assert V.verify_record(r, C.build_executors(r, by_dom)) != []


def test_unknown_alias_is_rejected(data):
    L, C, by_dom, lib, q, recs = data
    r = recs[0]
    ex = C.build_executors(r, by_dom)["R1"]
    native, rej, flags = ex.execute([("definitely_not_a_tool", {})])
    assert native == [] and (rej or flags)


def test_alias_names_never_collide_with_native(data):
    L, C, by_dom, lib, q, recs = data
    for dom, tools in by_dom.items():
        al, amap = L.make_alias_tools(tools)
        names = [t["name"] for t in tools] + [t["name"] for t in al]
        assert len(names) == len(set(names))


def test_deterministic_rebuild(data):
    L, C, by_dom, lib, q, recs = data
    r0 = recs[3]
    qq = next(x for x in q if x["id"] == r0["task_id"])
    r1, _ = C.build_record(qq, by_dom, lib, 0)
    assert L.h(r1) == L.h(r0)


def test_no_text_leakage_between_behaviors(data):
    """A control behavior's text must never equal any G variant's text (no label leakage via duplicates)."""
    L, C, by_dom, lib, q, recs = data
    for r in recs:
        gt = {x["text"] for x in r["responses"] if x["behavior"] == "G"}
        assert not any(x["text"] in gt for x in r["responses"] if x["behavior"] != "G")
