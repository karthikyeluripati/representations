"""Uncertain-outcome fault episodes on tau2-bench airline (vendored @c06b07dd, third_party/tau2_c06b07dd; MIT).

Zero-shot transfer environment: nothing from tau2 is used for training. Episode design mirrors the BFCL suites:
  * history = the task's customer request (reason_for_call + known_info + task_instructions, rewritten to first
    person) and the gold actions BEFORE the faulted write, executed as the agent's own calls (forced prefix);
  * the faulted write w is issued as the agent's call and returns an uncertain-outcome error:
        ACa: w was applied       ACl: w was not applied        (same observation)
    S0 = no fault: the prefix stops before w and the agent must issue w itself;
  * the agent then continues; success = final DB hash equals the gold end state (all gold actions from the initial
    state) and the agent did not escalate / transfer to a human.
Requires pydantic>=2.12, loguru, docstring-parser, deepdiff, addict, python-dotenv, toml (see README_VENDOR.md).
"""
from __future__ import annotations

import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
T2 = os.path.abspath(os.path.join(HERE, "..", "..", "third_party", "tau2_c06b07dd"))
ESCALATE = "escalate_to_user"
HUMAN = "transfer_to_human_agents"
_LOADED = {}


def _load():
    if _LOADED:
        return _LOADED
    os.environ.setdefault("TAU2_DATA_DIR", os.path.join(T2, "data"))
    if "tau2" not in sys.modules:                       # skip tau2/__init__.py (imports the LLM runner stack)
        pkg = types.ModuleType("tau2"); pkg.__path__ = [os.path.join(T2, "src", "tau2")]; sys.modules["tau2"] = pkg
    from loguru import logger
    logger.remove()
    from tau2.domains.airline.data_model import FlightDB
    from tau2.domains.airline.environment import get_environment, get_tasks
    from tau2.domains.airline.utils import AIRLINE_DB_PATH
    _LOADED.update(get_environment=get_environment, tasks={t.id: t for t in get_tasks(None)}, base=FlightDB.load(AIRLINE_DB_PATH),
                   policy=open(os.path.join(T2, "data", "tau2", "domains", "airline", "policy.md"), encoding="utf-8").read())
    return _LOADED


def fresh_env(task):
    L = _load()
    env = L["get_environment"](db=L["base"])
    if task.initial_state and task.initial_state.initialization_actions:
        for a in task.initial_state.initialization_actions:
            env.run_env_function_call(a)
    return env


def _obs(r):
    if isinstance(r, str):
        return r
    try:
        return json.dumps(r.model_dump(mode="json") if hasattr(r, "model_dump") else
                          [x.model_dump(mode="json") if hasattr(x, "model_dump") else x for x in r] if isinstance(r, list) else r)
    except Exception:  # noqa: BLE001
        return str(r)


def execute(env, name, args):
    try:
        return _obs(env.use_tool(name, **args))
    except Exception as e:  # noqa: BLE001  (tau2 tools raise ValueError for invalid requests: return them as errors)
        return f"Error: {e}"


def first_person(s):
    for a, b in ((r"\bYou are\b", "I am"), (r"\byou are\b", "I am"), (r"\bYou\b", "I"), (r"\byou\b", "me"), (r"\bYour\b", "My"),
                 (r"\byour\b", "my"), (r"\bdo not\b", "do not")):
        s = re.sub(a, b, s)
    return s


def customer_request(task):
    ins = task.user_scenario.instructions
    parts = [getattr(ins, k, None) for k in ("reason_for_call", "known_info", "task_instructions")]
    return "Customer request: " + first_person(" ".join(p.strip() for p in parts if p))


def tools(env):
    ts = [t.openai_schema for t in env.get_tools()]
    ts.append({"type": "function", "function": {"name": ESCALATE, "description": "Stop working on the request and tell the user it cannot be completed right now, with the reason.",
                                                "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Why the request cannot be completed."}}, "required": ["reason"]}}})
    return ts


_GOLD = {}


def gold_hash(task):
    if task.id not in _GOLD:                          # deterministic per task: cache
        env = fresh_env(task)
        for a in task.evaluation_criteria.actions:
            execute(env, a.name, a.arguments)
        _GOLD[task.id] = env.get_db_hash()
    return _GOLD[task.id]


class Tau2Episode:
    """scenario in {S0, ACa, ACl}; event = (task_id, index of the faulted gold write)."""

    def __init__(self, task_id, idx, scenario, ac_obs, max_calls=20):
        L = _load()
        self.task = L["tasks"][task_id]; self.idx = idx; self.scenario = scenario
        self.env = fresh_env(self.task)
        acts = self.task.evaluation_criteria.actions
        self.calls, self.escalated, self.done, self.max_calls = [], False, False, max_calls
        for a in acts[:idx]:
            self.calls.append((a.name, dict(a.arguments), execute(self.env, a.name, a.arguments)))
        if scenario in ("ACa", "ACl"):
            w = acts[idx]
            if scenario == "ACa":
                execute(self.env, w.name, w.arguments)
            self.calls.append((w.name, dict(w.arguments), ac_obs))
        self.n_forced = len(self.calls)
        self.target = acts[idx].name

    def call(self, name, args):
        if self.done:
            return "Error: episode finished"
        if name in (ESCALATE, HUMAN):
            self.escalated = self.done = True
            self.calls.append((name, args, "ESCALATED")); return "Escalated to the user."
        o = execute(self.env, name, args)
        self.calls.append((name, args, o))
        if len(self.calls) - self.n_forced >= self.max_calls:
            self.done = True
        return o

    def finish(self):
        self.done = True

    def grade(self):
        ok = (not self.escalated) and self.env.get_db_hash() == gold_hash(self.task)
        return {"scenario": self.scenario, "correct": bool(ok), "escalated": self.escalated, "n_calls": len(self.calls) - self.n_forced}

    def messages(self, system):
        msgs = [{"role": "system", "content": system + "\n\n# Airline policy\n" + _load()["policy"]},
                {"role": "user", "content": customer_request(self.task)}]
        for n, a, o in self.calls:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}}]})
            msgs.append({"role": "tool", "content": o[:2000]})
        return msgs


def events(path=os.path.join(HERE, "..", "..", "results", "tau2", "airline_write_events.json")):
    rows = json.load(open(path, encoding="utf-8"))
    return {"diagnostic": [(r["task"], r["i"], r["fn"]) for r in rows if r["non_idempotent"] and r["read_shows_change"]],
            "control": [(r["task"], r["i"], r["fn"]) for r in rows if not r["non_idempotent"]]}
