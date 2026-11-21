"""Tests for the WebState simulator and pair labels (docs/strong_paper/benchmark_protocol.md)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "strong_paper"))
import novelty_reliability as NR  # noqa: E402
import webstate as W  # noqa: E402


def test_render_deterministic_across_calls():
    fam = W.FAMILIES["settings"]
    cfg = W.RenderCfg(layout_seed=5, decor=["x"], style=True, label_var=1)
    a = W.render_history(fam, {"role": "user", "quota_full": False}, ["settings_link", "notif"], cfg)[0]
    b = W.render_history(fam, {"role": "user", "quota_full": False}, ["settings_link", "notif"], cfg)[0]
    assert a == b


def test_cosmetic_ops_preserve_state_and_change_text():
    fam = W.FAMILIES["shop"]
    hid = {"member": True, "in_stock": True}
    seq = ["product_link", "add_btn"]
    base = NR.make_item(fam, hid, seq, W.RenderCfg())
    for cfg in (W.RenderCfg(decor=["Banner"]), W.RenderCfg(style=True), W.RenderCfg(label_var=2)):
        it = NR.make_item(fam, hid, seq, cfg)
        assert it["state"] == base["state"]
        assert it["obs"][-1] != base["obs"][-1]


def test_false_equivalence_example_verified_by_execution():
    # same rendered home page; unsaved draft vs saved -> logout behaves differently
    fam = W.FAMILIES["settings"]
    hid = {"role": "user", "quota_full": False}
    i1 = NR.make_item(fam, hid, ["settings_link", "notif", "home_link"], W.RenderCfg())
    i2 = NR.make_item(fam, hid, ["settings_link", "notif", "save_btn", "home_link"], W.RenderCfg())
    assert i1["obs"][-1] == i2["obs"][-1]
    eq, wit = W.h_equivalent(fam, i1["state"], i2["state"], 3)
    assert not eq and wit is not None
    o1 = W.sem_obs(fam, W.step(fam, i1["state"], "logout_btn"))
    o2 = W.sem_obs(fam, W.step(fam, i2["state"], "logout_btn"))
    assert o1 != o2  # confirm dialog vs signed out
    assert NR.identifiability(fam, i1, i2)[0] == "determined"


def test_identical_history_hidden_difference_is_ambiguous():
    fam = W.FAMILIES["settings"]
    seq = ["settings_link", "home_link"]
    i1 = NR.make_item(fam, {"role": "user", "quota_full": False}, seq, W.RenderCfg())
    i2 = NR.make_item(fam, {"role": "admin", "quota_full": False}, seq, W.RenderCfg())
    assert i1["obs"] == i2["obs"]
    assert not W.h_equivalent(fam, i1["state"], i2["state"], 3)[0]
    ident, p = NR.identifiability(fam, i1, i2)
    assert ident == "ambiguous" and 0 < p < 1
    # every deployment-side key is a function of the observable history -> cannot separate
    for f in NR.KEYS.values():
        assert f(i1) == f(i2)


def test_equivalent_route_is_h_equivalent():
    fam = W.FAMILIES["settings"]
    hid = {"role": "user", "quota_full": False}
    a = NR.make_item(fam, hid, ["settings_link"], W.RenderCfg())
    b = NR.make_item(fam, hid, ["menu_btn", "settings_link"], W.RenderCfg())
    assert W.h_equivalent(fam, a["state"], b["state"], 5)[0]
    assert NR.KEYS["hist_full"](a) != NR.KEYS["hist_full"](b)  # history key -> false novelty
    assert NR.KEYS["obs_canon"](a) == NR.KEYS["obs_canon"](b)


def test_delayed_effect_mail():
    fam = W.FAMILIES["mail"]
    hid = {"offline": False}
    s = W.run(fam, hid, ["compose_link", "fill_to", "send_later_btn"])[-1]
    assert s["sent"] == 0
    for _ in range(3):
        s = W.step(fam, s, "inbox_link" if s["page"] != "inbox" else "compose_link")
    assert s["sent"] == 1


def test_hidden_variable_names_never_rendered():
    rng = random.Random(0)
    for fam in W.FAMILIES.values():
        for it in NR.sample_items(fam, 60, rng):
            txt = "\n".join(it["obs"]).lower()
            # 'role' is deliberately exposed by a visible field on the account page (a probe-able variable)
            for k in fam.HIDDEN:
                if k != "role":
                    assert k.lower() not in txt


def test_invalid_action_rejected():
    fam = W.FAMILIES["ticket"]
    s = fam.init({"approve_delay": 2, "can_escalate": False})
    try:
        W.step(fam, s, "escalate_btn")
    except ValueError:
        return
    raise AssertionError("unavailable action executed")
