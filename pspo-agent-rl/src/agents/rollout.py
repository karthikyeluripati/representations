"""Batched multi-step tool-agent rollouts with vLLM on the paired-fault BFCL episodes.

The policy sees only: system prompt, earlier turns (executed ground truth, shown as history), the current user request,
tool schemas (identical across scenarios) and tool observations. Hidden scenario labels are never in the prompt.
"""
from __future__ import annotations

import copy
import json
import re

from environments.bfcl_faults import Episode, ERRORS, new_instances, parse_call, run_calls, tool_schemas

SYSTEM_BASE = "You are a helpful assistant that completes user requests by calling the provided tools."
SYSTEM_RECOVERY = SYSTEM_BASE + (
    " Tool calls can fail. If a call fails, you may retry it; if a function keeps failing, use its backup endpoint if one "
    "exists; if every way to complete the request keeps failing, call escalate_to_user instead of continuing. Do not give up "
    "on a request that can still be completed.")
TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
MAX_TURNS = 8


def _convert(p):
    if isinstance(p, dict):
        out = {}
        for k, v in p.items():
            if k == "type" and isinstance(v, str):
                v = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}.get(v, v)
            out[k] = _convert(v)
        return out
    if isinstance(p, list):
        return [_convert(x) for x in p]
    return p


def openai_tools(task, docs):
    return [{"type": "function", "function": {"name": d["name"], "description": d.get("description", ""),
                                              "parameters": _convert(d.get("parameters", {"type": "object", "properties": {}}))}}
            for d in tool_schemas(task, docs)]


def initial_messages(task, system):
    msgs = [{"role": "system", "content": system}]
    inst = new_instances(task["entry"])
    for h in task["history"]:
        msgs.append({"role": "user", "content": h["user"]})
        calls = [parse_call(c) for c in h["calls"]]
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]})
        for c in h["calls"]:
            r = run_calls(inst, [c])[0]
            msgs.append({"role": "tool", "content": r if isinstance(r, str) else json.dumps(r, default=str)})
        msgs.append({"role": "assistant", "content": "Done."})
    msgs.append({"role": "user", "content": task["user"]})
    return msgs


def parse_tool_calls(text):
    calls = []
    for m in TC_RE.finditer(text):
        try:
            d = json.loads(m.group(1))
            args = d.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append((d["name"], args if isinstance(args, dict) else {}))
        except Exception:  # noqa: BLE001
            calls.append(("__malformed__", {}))
    return calls


class Runner:
    def __init__(self, llm, tokenizer, docs, system=SYSTEM_BASE, rule_max_consecutive_errors=None):
        self.llm, self.tok, self.docs, self.system = llm, tokenizer, docs, system
        self.rule_n = rule_max_consecutive_errors

    def render(self, msgs, tools):
        return self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)

    def run(self, items, sampling):
        """items: list of (task, scenario, seed). Returns list of trajectory dicts."""
        from vllm import SamplingParams  # noqa: F401
        st = []
        for task, sc, seed in items:
            tools = openai_tools(task, self.docs)
            st.append({"task": task, "scenario": sc, "seed": seed, "tools": tools, "ep": Episode(task, sc, seed),
                       "msgs": initial_messages(task, self.system), "steps": [], "done": False, "consec_err": 0,
                       "tok_in": 0, "tok_out": 0, "forced_escalation": False})
        for _ in range(MAX_TURNS):
            act = [s for s in st if not s["done"]]
            if not act:
                break
            prompts = [self.render(s["msgs"], s["tools"]) for s in act]
            outs = self.llm.generate(prompts, sampling, use_tqdm=False)
            for s, p, o in zip(act, prompts, outs):
                text = o.outputs[0].text
                s["tok_in"] += len(o.prompt_token_ids); s["tok_out"] += len(o.outputs[0].token_ids)
                calls = parse_tool_calls(text)
                s["steps"].append({"prompt": p, "completion": text, "n_calls": len(calls)})
                content = TC_RE.sub("", text).strip()
                if not calls:
                    s["msgs"].append({"role": "assistant", "content": content}); s["done"] = True; continue
                s["msgs"].append({"role": "assistant", "content": content,
                                  "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]})
                for n, a in calls:
                    obs = s["ep"].call(n, a) if n != "__malformed__" else "Error: malformed tool call"
                    s["msgs"].append({"role": "tool", "content": obs[:2000]})
                    s["consec_err"] = s["consec_err"] + 1 if (obs in ERRORS or obs.startswith("Error")) else 0
                    if s["ep"].done:
                        break
                    if self.rule_n and s["consec_err"] >= self.rule_n:   # B1: bounded-retry rule wrapper
                        s["ep"].call("escalate_to_user", {"reason": "repeated tool failures"}); s["forced_escalation"] = True
                        break
                if s["ep"].done:
                    s["done"] = True
        res = []
        for s in st:
            g = s["ep"].grade()
            res.append({"task_id": s["task"]["id"], "base_id": s["task"]["base_id"], "split": s["task"]["split"], "fault_class": s["task"]["fault_class"],
                        "scenario": s["scenario"], "seed": s["seed"], **g, "tok_in": s["tok_in"], "tok_out": s["tok_out"],
                        "forced_escalation": s["forced_escalation"], "calls": [(n, a, o[:300]) for n, a, o in s["ep"].calls],
                        "steps": s["steps"]})
        return res
