"""Trainer behaviour on a tiny model (learning signal, anchor effect, gradient modes) and evaluation plumbing."""
import contextlib
import copy

import torch

from src.evaluation import metrics as M
from src.evaluation.agent_loop import ScriptedPolicy, parse_tool_calls, run
from src.training import objectives as O
from src.training.render import ToyRenderer
from src.training.trainer import Encoder, Trainer, seq_logps

TOOLS = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in ("fix", "retry")]


def _call(n):
    return {"name": n, "arguments": {"x": 1}}


def _state(obs):
    return [{"role": "user", "content": "do it"}, {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": _call("go")}]},
            {"role": "tool", "content": obs}]


def _setup(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(Qwen3Config(vocab_size=258, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, head_dim=16, tie_word_embeddings=True))
    r = ToyRenderer()
    enc = Encoder(r, lambda tid: TOOLS, 4096)
    frozen = copy.deepcopy(m).eval()

    @contextlib.contextmanager
    def ref():
        live = [p.data for p in m.parameters()]
        for p, q in zip(m.parameters(), frozen.parameters()):
            p.data = q.data
        try:
            yield
        finally:
            for p, d in zip(m.parameters(), live):
                p.data = d
    return m, r, enc, ref


def _train(anchor, mode="sum", steps=30):
    m, r, enc, ref = _setup()
    pairs = enc.encode_all([{"type": "pair", "task_id": "t", "msgs": _state(f"Error: real {i}"), "chosen": _call("fix"), "rejected": _call("retry")} for i in range(4)])
    anchors = enc.encode_all([{"type": "anchor", "task_id": "t", "msgs": _state(f"Error: outage {i}"), "action": _call("retry")} for i in range(4)]) if anchor else []
    cfg = {"objective": "dpo", "lr": 3e-3, "beta": 0.5, "batch_size": 4, "anchor_batch_size": 4, "micro_batch": 2, "steps": steps,
           "lambda": 1.0, "grad_mode": mode, "anchor": "nll" if anchor else None, "warmup_frac": 0.0}
    tr = Trainer(m, ref, r.pad_id, cfg, pairs, anchors)
    tr.precompute_ref()
    probe = enc.encode_all([{"type": "anchor", "task_id": "t", "msgs": _state("Error: outage 9"), "action": _call("retry")}])
    with torch.no_grad():
        before = float(seq_logps(m, probe, r.pad_id, "cpu")[0])
    hist = tr.train()
    with torch.no_grad():
        after = float(seq_logps(m, probe, r.pad_id, "cpu")[0])
    return hist, after - before


def test_dpo_learns_and_anchor_limits_collateral_drift():
    h0, drift0 = _train(anchor=False)
    h1, drift1 = _train(anchor=True)
    assert h0[-1]["dpo_margin"] > h0[0]["dpo_margin"] and h0[-1]["dpo_acc"] == 1.0
    assert h1[-1]["dpo_acc"] == 1.0
    assert drift1 > drift0                     # the anchored run keeps more probability on the protected action
    assert "cos_PA" in h1[-1]


def test_gradient_modes_run():
    for mode in ("pcgrad", "agem"):
        h, _ = _train(anchor=True, mode=mode, steps=3)
        assert len(h) == 3


def test_parse_tool_calls():
    t = '<tool_call>\n{"name": "cd", "arguments": {"folder": "a"}}\n</tool_call> x <tool_call>{bad}</tool_call>'
    assert parse_tool_calls(t) == [("cd", {"folder": "a"}), ("__malformed__", {})]


def test_false_outage_events():
    err = '{"error": "No user is currently logged in."}'
    row = {"calls": [("send", {"a": 1}, err), ("send", {"a": 1}, err), ("escalate_to_user", {}, "ESCALATED")], "n_forced": 1}
    assert M.false_outage_events(row) == [True, True]
    row2 = {"calls": [("send", {"a": 1}, err), ("login", {}, "ok"), ("send", {"a": 1}, "ok")], "n_forced": 1}
    assert M.false_outage_events(row2) == [False]


def test_scripted_oracle_solves_sample_and_paired_diff_zero(tasks, by_id):
    from src.environment.bfcl_env import load_bfcl, openai_tools
    _, _, docs = load_bfcl()
    T = [t for t in tasks if t["split"] == "heldout"][:3]
    specs = [{"task_id": t["id"], "pool": "v2", "alt": "train", "suite": sc, "scenario": sc, "seed": 0} for t in T for sc in ("S0", "S1", "S2", "S3")]
    rows = run(specs, ScriptedPolicy("oracle", by_id), by_id, lambda tid, alt: openai_tools(by_id[tid], docs))
    assert all(r["correct"] for r in rows)
    p, ci = M.paired_diff(rows, rows, M.metric_fn("overall"), B=50)
    assert p == 0 and ci == (0.0, 0.0)
