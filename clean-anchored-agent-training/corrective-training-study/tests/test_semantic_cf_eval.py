"""CPU test of the batched model-evaluation loop: a scripted policy that emits <tool_call> text must reproduce
run_policy exactly, and malformed / unknown calls must be graded, not crash."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import eval_semantic_cf as E  # noqa: E402
from verification import semantic_cf as S  # noqa: E402


class TextPolicy:
    def __init__(self, inner):
        self.inner = inner

    def act(self, states):
        out = []
        for s in states:
            a = self.inner.act(s["msgs"], s["tools"])
            out.append(a["final_text"] if "final_text" in a else
                       f"<tool_call>\n{json.dumps({'name': a['name'], 'arguments': a.get('arguments', {})})}\n</tool_call>")
        return out


class Garbage:
    def act(self, states):
        return ["<tool_call>{not json}</tool_call>" if i % 2 else '<tool_call>{"name": "nope", "arguments": {}}</tool_call>'
                for i, _ in enumerate(states)]


SP = S.specs("test")[:400]


def test_batch_loop_matches_scripted_runner():
    for P in (S.ContractPolicy, S.AlwaysRetry, lambda: S.CheckFirst(True)):
        a = E.run_batch(TextPolicy(P()), SP); b = S.run_policy(P(), SP, S.SYSTEM)
        key = ("correct", "duplicate", "escalated", "reissues", "checks", "first", "n_calls")
        assert [{k: r[k] for k in key} for r in a] == [{k: r[k] for k in key} for r in b]


def test_malformed_calls_are_graded():
    rows = E.run_batch(Garbage(), SP)
    assert len(rows) == len(SP) and all(r["n_calls"] <= S.CFEpisode.MAX_CALLS for r in rows)
    assert not any(r["correct"] for r in rows if not r["feasible"])
