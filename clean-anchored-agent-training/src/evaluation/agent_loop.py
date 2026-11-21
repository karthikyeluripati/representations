"""Multi-step tool-agent loop over BFCL episodes, shared by GPU evaluation and CPU grader validation.

A policy maps a batch of live episode states to completions in Qwen3 tool-call format
    <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
  * VLLMPolicy    renders the prompt with the model's chat template and generates (GPU).
  * ScriptedPolicy emits a precomputed action plan (CPU; validates parsing, execution and grading end-to-end).
The loop ends when the model replies without a tool call, escalates, or the call/step budget is reached.
"""
from __future__ import annotations

import json
import re

from src.counterfactuals.construct import episode_messages
from src.counterfactuals.policies import OutageRule, Repairer
from src.environment.bfcl_env import EVAL_ALT, EVAL_ERRORS, MAX_TOOL_CALLS, TRAIN_ALT, V2_ERRORS, Episode

TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
POOLS = {"v2": V2_ERRORS, "eval": EVAL_ERRORS}
ALTS = {"train": TRAIN_ALT, "eval": EVAL_ALT}
MAX_STEPS = 20


def parse_tool_calls(text):
    calls = []
    for m in TC_RE.finditer(text):
        try:
            d = json.loads(m.group(1))
            a = d.get("arguments", {})
            a = json.loads(a) if isinstance(a, str) else a
            calls.append((d["name"], a if isinstance(a, dict) else {}))
        except Exception:  # noqa: BLE001
            calls.append(("__malformed__", {}))
    return calls


def fmt_call(name, args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def make_episode(spec, by_id):
    t = by_id[spec["task_id"]]
    return Episode(t, spec["scenario"], spec["seed"], error_pool=POOLS[spec["pool"]], alt=ALTS[spec["alt"]],
                   forced=spec.get("forced"), max_tool_calls=MAX_TOOL_CALLS + len(spec.get("forced") or []),
                   ac_msg=spec.get("ac_msg", "explicit"))


class ScriptedPolicy:
    """kind='oracle': ground truth + outage procedure, repairs real errors via the known repair plan.
    kind='surrogate': outage procedure applied to every error (the over-generalising policy)."""

    def __init__(self, kind, by_id):
        self.kind, self.by_id = kind, by_id

    def plan(self, st):
        ep = st["ep"].snapshot()
        n0 = len(ep.calls)
        spec = st["spec"]
        if spec["suite"] in ("RP", "RA"):
            last = ep.calls[-1]
            pol = Repairer(spec["repair_plan"]) if self.kind == "oracle" else OutageRule()
            pending = spec["repair_plan"] if self.kind == "oracle" else spec["naive_rest"]
            if self.kind == "oracle":
                pol.continue_from(ep, pending)
            else:
                pol.continue_from(ep, pending, last=(last[0], last[1], last[2]))
        else:
            OutageRule().continue_from(ep, self.by_id[spec["task_id"]]["turn_gt"])
        return [fmt_call(n, a) for n, a, _ in ep.calls[n0:]] + ([] if ep.escalated else ["Done."])

    def act(self, states):
        out = []
        for st in states:
            if "plan" not in st:
                st["plan"] = self.plan(st)
            out.append(st["plan"].pop(0) if st["plan"] else "Done.")
        return out


class VLLMPolicy:
    def __init__(self, llm, tokenizer, sampling, lora_request=None, template_kwargs=None):
        self.llm, self.tok, self.sp, self.lora = llm, tokenizer, sampling, lora_request
        self.tkw = template_kwargs or {}                     # e.g. {"enable_thinking": False} for hybrid Qwen3 models

    def act(self, states):
        prompts = [self.tok.apply_chat_template(s["msgs"], tools=s["tools"], add_generation_prompt=True, tokenize=False, **self.tkw) for s in states]
        outs = self.llm.generate(prompts, self.sp, use_tqdm=False, lora_request=self.lora)
        for s, o in zip(states, outs):
            s["tok_in"] = s.get("tok_in", 0) + len(o.prompt_token_ids); s["tok_out"] = s.get("tok_out", 0) + len(o.outputs[0].token_ids)
        return [o.outputs[0].text for o in outs]


def run(specs, policy, by_id, tools_fn, max_steps=MAX_STEPS, system=None):
    states = []
    for sp in specs:
        ep = make_episode(sp, by_id)
        sysm = sp.get("system") or system
        msgs = episode_messages(by_id[sp["task_id"]], ep, sysm) if sysm else episode_messages(by_id[sp["task_id"]], ep)
        states.append({"spec": sp, "ep": ep, "msgs": msgs,
                       "tools": tools_fn(sp["task_id"], sp["alt"]), "done": False, "steps": [], "n_model_calls": 0})
    for _ in range(max_steps):
        act = [s for s in states if not s["done"]]
        if not act:
            break
        for s, text in zip(act, policy.act(act)):
            s["n_model_calls"] += 1
            calls = parse_tool_calls(text)
            s["steps"].append(text[:4000])
            content = TC_RE.sub("", text).strip()
            if not calls:
                s["msgs"].append({"role": "assistant", "content": content}); s["done"] = True
                continue
            s["msgs"].append({"role": "assistant", "content": content,
                              "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]})
            for n, a in calls:
                obs = s["ep"].call(n, a) if n != "__malformed__" else "Error: malformed tool call"
                s["msgs"].append({"role": "tool", "content": obs[:2000]})
                if s["ep"].done:
                    break
            if s["ep"].done:
                s["done"] = True
    res = []
    for s in states:
        g = s["ep"].grade()
        sp = s["spec"]
        t = by_id[sp["task_id"]]
        res.append({**{k: v for k, v in sp.items() if k not in ("forced", "repair_plan", "naive_rest")}, "base_id": t["base_id"],
                    "split": t["split"], **g, "hit_step_cap": not s["done"],
                    "calls": [(n, a, o[:300]) for n, a, o in s["ep"].calls], "n_forced": s["ep"].n_forced,
                    "n_model_calls": s["n_model_calls"], "tok_in": s.get("tok_in", 0), "tok_out": s.get("tok_out", 0),
                    "completions": s["steps"]})
    return res
