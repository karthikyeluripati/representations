"""Paired recoverability scenarios on BFCL v4 multi-turn executable backends (Apache-2.0, bfcl-eval 2026.3.23).

Episode = the FIRST user turn of a BFCL `multi_turn_base` task, executed against BFCL's real tool classes, graded with
BFCL's own state comparison (all public attributes of every involved instance must equal the ground-truth instance).

Fault target f = the first ground-truth call of turn 0 that changes the state of its class C (determined by execution).
Every scenario exposes the SAME tools: all BFCL functions of the involved classes, a backup endpoint `<f>_backup`
(identical signature/semantics), and `escalate_to_user(reason)`. The prompt is identical across scenarios.

  S0 (control): no faults.
  S1 (retry viable): the first k in {1,2} calls to f fail; later calls succeed. The backup also works.
  S2 (switch needed): f always fails; `<f>_backup` works.
  S3 (impossible): every method of class C, including the backup, fails for the whole episode. C's state can only change
      through its methods, and the ground truth requires C's state to change => completion is impossible (verified).
Error text is drawn from ONE shared pool for all scenarios, so a single error message does not reveal the scenario;
the scenarios become distinguishable only through further interaction (retry outcome, backup outcome, other C calls).
"""
from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
import os
import random

import bfcl_eval
from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES

B = os.path.dirname(bfcl_eval.__file__)
DOC_FILES = {"GorillaFileSystem": "gorilla_file_system.json", "MathAPI": "math_api.json", "MessageAPI": "message_api.json",
             "TwitterAPI": "posting_api.json", "TicketAPI": "ticket_api.json", "TradingBot": "trading_bot.json",
             "TravelAPI": "travel_booking.json", "VehicleControlAPI": "vehicle_control.json"}
ERRORS = ["Error: service temporarily unavailable (HTTP 503). Please try again later.",
          "Error: request timed out after 30s.",
          "Error: upstream connection reset by peer."]
MAX_TOOL_CALLS = 12


def load_bfcl():
    q = [json.loads(l) for l in open(os.path.join(B, "data", "BFCL_v4_multi_turn_base.json"))]
    a = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(os.path.join(B, "data", "possible_answer", "BFCL_v4_multi_turn_base.json"))}
    docs = {}
    for cls, fn in DOC_FILES.items():
        docs[cls] = [json.loads(l) for l in open(os.path.join(B, "data", "multi_turn_func_doc", fn))]
    return q, a, docs


def new_instances(entry):
    inst = {}
    cfg = entry["initial_config"]
    for cls in entry["involved_classes"]:
        mod = importlib.import_module(CLASS_FILE_PATH_MAPPING[cls])
        obj = getattr(mod, cls)()
        if cls not in STATELESS_CLASSES:
            obj._load_scenario(copy.deepcopy(cfg.get(cls, {})))
        inst[cls] = obj
    return inst


def public_state(obj):
    return {k: copy.deepcopy(v) for k, v in vars(obj).items() if not k.startswith("_")}


def method_owner(inst, name):
    for cls, obj in inst.items():
        m = getattr(obj, name, None)
        if m is not None and inspect.ismethod(m) and not name.startswith("_"):
            return cls
    return None


def parse_call(s):
    """'mv(source="a", destination="b")' -> ("mv", {...}) using the Python AST (BFCL ground-truth format)."""
    node = ast.parse(s.strip(), mode="eval").body
    name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    return name, {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def run_calls(inst, calls):
    out = []
    for c in calls:
        name, args = parse_call(c)
        cls = method_owner(inst, name)
        try:
            out.append(getattr(inst[cls], name)(**args))
        except Exception as e:  # noqa: BLE001
            out.append(f"Error: {e}")
    return out


def _user_text(entry, j):
    q = entry["question"][j]
    q = q if isinstance(q, list) else [q]
    users = [m["content"] for m in q if m.get("role") == "user"]
    return users[-1] if users else None


def build_task(entry, gt, turn=0):
    """Episode = user turn `turn`. Earlier ground-truth turns are executed (state prefix) and shown as history.
    Returns a task dict or None if the turn has no user text or no state-changing call."""
    if turn >= len(gt) or not gt[turn] or _user_text(entry, turn) is None:
        return None
    prefix = [c for t in gt[:turn] for c in t]
    inst = new_instances(entry); run_calls(inst, prefix)
    for c in gt[turn]:
        name, args = parse_call(c)
        cls = method_owner(inst, name)
        if cls is None or cls in STATELESS_CLASSES:
            run_calls(inst, [c]); continue
        before = public_state(inst[cls])
        run_calls(inst, [c])
        if public_state(inst[cls]) != before:
            start = new_instances(entry); run_calls(start, prefix)
            final = new_instances(entry); run_calls(final, prefix + list(gt[turn]))
            if public_state(final[cls]) == public_state(start[cls]):
                return None
            history = []
            for j in range(turn):
                if _user_text(entry, j) is None:
                    return None
                history.append({"user": _user_text(entry, j), "calls": list(gt[j])})
            return {"id": f"{entry['id']}#t{turn}", "base_id": entry["id"], "turn": turn, "entry": entry, "prefix_gt": prefix,
                    "turn_gt": list(gt[turn]), "fault_fn": name, "fault_class": cls, "user": _user_text(entry, turn), "history": history}
    return None


def tool_schemas(task, docs):
    e = task["entry"]
    fns = []
    excluded = set(e.get("excluded_function", []))
    for cls in e["involved_classes"]:
        for d in docs.get(cls, []):
            if d["name"] not in excluded:
                fns.append(d)
    f = next(d for d in fns if d["name"] == task["fault_fn"])
    bk = copy.deepcopy(f)
    bk["name"] = f["name"] + "_backup"
    bk["description"] = f"Backup endpoint for {f['name']} (same arguments and effect). " + f["description"]
    esc = {"name": "escalate_to_user", "description": "Stop working on the request and tell the user it cannot be completed right now, with the reason.",
           "parameters": {"type": "dict", "properties": {"reason": {"type": "string", "description": "Why the request cannot be completed."}}, "required": ["reason"]}}
    return fns + [bk, esc]


class Episode:
    def __init__(self, task, scenario, seed):
        self.task, self.scenario = task, scenario
        rng = random.Random(f"{task['id']}:{scenario}:{seed}")
        self.k = rng.choice([1, 2])                  # S1 transient failure count
        self.err = rng.choice(ERRORS)                 # same pool for every scenario
        self.inst = new_instances(task["entry"])
        run_calls(self.inst, task["prefix_gt"])
        self.calls, self.n_primary, self.escalated, self.done = [], 0, False, False
        self.first_escalation_call = None

    def call(self, name, args):
        if self.done:
            return "Error: episode finished"
        if name == "escalate_to_user":
            self.escalated, self.done = True, True
            self.first_escalation_call = len(self.calls)
            self.calls.append((name, args, "ESCALATED"))
            return "Escalated to the user."
        f, C = self.task["fault_fn"], self.task["fault_class"]
        target = f if name == f + "_backup" else name
        cls = method_owner(self.inst, target)
        if cls is None:
            obs = f"Error: unknown function {name}"
        else:
            fail = False
            if self.scenario == "S1" and name == f:
                self.n_primary += 1
                fail = self.n_primary <= self.k
            elif self.scenario == "S2" and name == f:
                fail = True
            elif self.scenario == "S3" and (cls == C):
                fail = True
            if fail:
                obs = self.err
            else:
                try:
                    obs = getattr(self.inst[cls], target)(**(args or {}))
                except Exception as e:  # noqa: BLE001
                    obs = f"Error: {type(e).__name__}: {e}"
        obs = obs if isinstance(obs, str) else json.dumps(obs, default=str)
        self.calls.append((name, args, obs))
        if len(self.calls) >= MAX_TOOL_CALLS:
            self.done = True
        return obs

    def grade(self):
        gt = new_instances(self.task["entry"])
        run_calls(gt, self.task["prefix_gt"] + self.task["turn_gt"])
        state_ok = all(public_state(self.inst[c]) == public_state(gt[c]) for c in gt)
        fault_calls = sum(1 for n, _, _ in self.calls if n in (self.task["fault_fn"], self.task["fault_fn"] + "_backup"))
        if self.scenario == "S3":
            correct = self.escalated
        else:
            correct = state_ok and not self.escalated
        return {"scenario": self.scenario, "correct": bool(correct), "escalated": self.escalated, "state_ok": state_ok,
                "n_calls": len(self.calls), "fault_fn_calls": fault_calls,
                "calls_before_escalation": self.first_escalation_call}


def reference_policy(ep):
    """Executable reference: replay turn-0 ground truth; on failure of f retry up to 2 more times, then switch to the
    backup (twice), then escalate. Used to verify solvability of S0-S2 and correct escalation in S3."""
    for c in ep.task["turn_gt"]:
        name, args = parse_call(c)
        obs = ep.call(name, args)
        if name == ep.task["fault_fn"] and obs in ERRORS:
            ok = False
            for _ in range(2):
                if ep.call(name, args) not in ERRORS:
                    ok = True; break
            if not ok:
                for _ in range(2):
                    if ep.call(name + "_backup", args) not in ERRORS:
                        ok = True; break
            if not ok:
                ep.call("escalate_to_user", {"reason": "service unavailable"}); return
        elif obs in ERRORS:  # another method of the down class (S3)
            ep.call("escalate_to_user", {"reason": "service unavailable"}); return
        if ep.done:
            return
