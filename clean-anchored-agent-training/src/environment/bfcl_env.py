"""BFCL v4 multi-turn episodes with controlled injected faults and preserved real API errors.

Built on the official BFCL executable backends (bfcl-eval 2026.3.23, Apache-2.0) and BFCL's state-based grading
(all public attributes of every involved class instance must equal the ground-truth end state).

Scenarios (one user turn per episode; earlier turns are executed ground truth shown as history):
  S0  no fault.
  S1  transient outage: the first k in {1,2} calls to the target f return an injected outage string.
  S2  persistent outage: f always fails; the alternative endpoint (default `<f>_backup`) works.
  S3  impossible: every method of f's class fails, alternative included -> correct behaviour is escalation.
  RP  real precondition error: a forced prefix in which the agent skipped a required ground-truth call, so a later
      call returns the REAL BFCL error (e.g. "No user is currently logged in"). Correct: repair and complete.
  ACa ambiguous commit, applied: the first call to f executes but returns AMBIGUOUS_COMMIT; re-issuing f duplicates it.
  ACl ambiguous commit, lost: the first call to f returns AMBIGUOUS_COMMIT without executing; f must be re-issued.
  RA  real argument error: a forced prefix in which one string argument has a one-character typo and the call returns
      the real BFCL error. Correct: re-issue with the right value (which is visible in context) and complete.
Injected outage strings come from an explicit pool; real errors are never rewritten. `obs_kind` separates them.
"""
from __future__ import annotations

import copy
import json
import os
import random

import bfcl_eval
from bfcl_eval.constants.executable_backend_config import STATELESS_CLASSES

from src.replay.replay import (execute, method_owner, new_instances, parse_call, bind_args, public_state, run_calls,
                               states_equal, to_obs)

BFCL_DIR = os.path.dirname(bfcl_eval.__file__)
DOC_FILES = {"GorillaFileSystem": "gorilla_file_system.json", "MathAPI": "math_api.json", "MessageAPI": "message_api.json",
             "TwitterAPI": "posting_api.json", "TicketAPI": "ticket_api.json", "TradingBot": "trading_bot.json",
             "TravelAPI": "travel_booking.json", "VehicleControlAPI": "vehicle_control.json"}

# Injected outage pools. V2 = the pool used by the previous project (kept for comparability).
# TRAIN / EVAL are disjoint; EVAL strings never appear in training data (observation-shift axis).
V2_ERRORS = ["Error: service temporarily unavailable (HTTP 503). Please try again later.",
             "Error: request timed out after 30s.",
             "Error: upstream connection reset by peer."]
TRAIN_ERRORS = ["Error: 503 Service Unavailable - the server is temporarily overloaded.",
                "Error: gateway timeout while contacting the service.",
                "Error: connection dropped by the remote host."]
EVAL_ERRORS = ["Error: backend unreachable (ECONNREFUSED).",
               "Error: deadline exceeded waiting for upstream.",
               "Error: the service is under maintenance, try again shortly."]
# Ambiguous-commit fault (ACa: the call WAS applied; ACl: it was NOT). Same string in both, so only verifying passes both.
AMBIGUOUS_COMMIT = "Error: request timed out after the server received it; the operation may or may not have been applied."
AC_MESSAGES = {"explicit": AMBIGUOUS_COMMIT,                                                  # says outcome is unknown
               "para": "Error: connection lost before a response arrived; the request might already have been processed.",
               "implicit": "Error: the request timed out.",                                   # realistic: says nothing
               # TRAIN-ONLY wordings for the OTV method study (frozen 2026-11-20; never used in evaluation):
               "train1": "Error: the response was lost after the request was sent; it is unknown whether the change was made.",
               "train2": "Error: upstream timeout after submission; the outcome of this operation could not be confirmed."}
# IAR study: the TRAIN outage strings of the timeout/connection type are, physically, ambiguous for a state-changing call
# (the request may have been processed). Registered here so ACa/ACl twins can use them; "503" stays a clean rejection.
AC_MESSAGES["t_gateway"] = TRAIN_ERRORS[1]
AC_MESSAGES["t_dropped"] = TRAIN_ERRORS[2]
AC_TRAIN_MSGS = ("explicit", "train1", "train2")
IAR_AMBIG_TRAIN_MSGS = ("t_gateway", "t_dropped")
IAR_REJECTION_TRAIN = TRAIN_ERRORS[0]
AC_EVAL_UNSEEN_MSGS = ("para", "implicit")
ALL_INJECTED = set(V2_ERRORS) | set(TRAIN_ERRORS) | set(EVAL_ERRORS) | set(AC_MESSAGES.values())
TRAIN_ALT = "{f}_backup"
EVAL_ALT = "{f}_replica"           # action-shift axis: the alternative endpoint has an unseen naming convention
ESCALATE = "escalate_to_user"
MAX_TOOL_CALLS = 20


def obs_kind(o: str) -> str:
    if o in ALL_INJECTED:
        return "injected"
    if o == "ESCALATED":
        return "escalate"
    lo = o.lower()
    if o.startswith("Error") or lo.startswith('{"error"') or '"error":' in lo[:40]:
        return "native"
    return "ok"


def load_bfcl():
    q = [json.loads(l) for l in open(os.path.join(BFCL_DIR, "data", "BFCL_v4_multi_turn_base.json"), encoding="utf-8")]
    a = {json.loads(l)["id"]: json.loads(l)["ground_truth"]
         for l in open(os.path.join(BFCL_DIR, "data", "possible_answer", "BFCL_v4_multi_turn_base.json"), encoding="utf-8")}
    docs = {cls: [json.loads(l) for l in open(os.path.join(BFCL_DIR, "data", "multi_turn_func_doc", fn), encoding="utf-8")]
            for cls, fn in DOC_FILES.items()}
    return q, a, docs


def _user_text(entry, j):
    q = entry["question"][j]
    q = q if isinstance(q, list) else [q]
    users = [m["content"] for m in q if m.get("role") == "user"]
    return users[-1] if users else None


def build_task(entry, gt, turn):
    """Episode = user turn `turn`; fault target = first ground-truth call of the turn that changes its class's state
    (measured by execution with the positional-argument-correct replay)."""
    if turn >= len(gt) or not gt[turn] or _user_text(entry, turn) is None:
        return None
    for j in range(turn):
        if _user_text(entry, j) is None:
            return None
    prefix = [c for t in gt[:turn] for c in t]
    inst = new_instances(entry); run_calls(inst, prefix)
    for c in gt[turn]:
        name = parse_call(c)[0]
        cls = method_owner(inst, name)
        if cls is None:
            return None                      # ground truth references a function we cannot resolve: drop the task
        before = public_state(inst[cls])
        run_calls(inst, [c])
        if cls in STATELESS_CLASSES or public_state(inst[cls]) == before:
            continue
        start = new_instances(entry); run_calls(start, prefix)
        final = new_instances(entry); run_calls(final, prefix + list(gt[turn]))
        if public_state(final[cls]) == public_state(start[cls]):
            return None
        history = [{"user": _user_text(entry, j), "calls": list(gt[j])} for j in range(turn)]
        return {"id": f"{entry['id']}#t{turn}", "base_id": entry["id"], "turn": turn, "entry": entry,
                "prefix_gt": prefix, "turn_gt": list(gt[turn]), "fault_fn": name, "fault_class": cls,
                "user": _user_text(entry, turn), "history": history}
    return None


_ANN = None


def _annotations():
    """MCP-style tool annotations (readOnlyHint / idempotentHint), derived by execution (scripts/label_idempotency.py).
    Enabled only when CTS_TOOL_ANNOTATIONS points to the labels file; otherwise tool docs are unchanged."""
    global _ANN
    p = os.environ.get("CTS_TOOL_ANNOTATIONS")
    if not p:
        return None
    if _ANN is None:
        _ANN = json.load(open(p, encoding="utf-8"))
    return _ANN


def annotate(d, ann):
    a = ann.get(d["name"]) if ann else None
    if not a:
        return d
    d = copy.deepcopy(d)
    d["description"] = (d.get("description", "") + f" [annotations: readOnlyHint={str(a['readOnlyHint']).lower()}, "
                        f"idempotentHint={str(a['idempotentHint']).lower()}]")
    return d


def tool_docs(task, docs, alt=TRAIN_ALT):
    e = task["entry"]
    excluded = set(e.get("excluded_function", []))
    ann = _annotations()
    fns = [annotate(d, ann) for cls in e["involved_classes"] for d in docs.get(cls, []) if d["name"] not in excluded]
    f = next(d for d in fns if d["name"] == task["fault_fn"])
    bk = copy.deepcopy(f)
    bk["name"] = alt.format(f=f["name"])
    bk["description"] = f"Alternative endpoint for {f['name']} (same arguments and effect). " + f.get("description", "")
    esc = {"name": ESCALATE, "description": "Stop working on the request and tell the user it cannot be completed right now, with the reason.",
           "parameters": {"type": "dict", "properties": {"reason": {"type": "string", "description": "Why the request cannot be completed."}},
                          "required": ["reason"]}}
    return fns + [bk, esc]


def _convert(p):
    if isinstance(p, dict):
        return {k: ({"dict": "object", "float": "number", "tuple": "array", "any": "string"}.get(v, v)
                    if k == "type" and isinstance(v, str) else _convert(v)) for k, v in p.items()}
    if isinstance(p, list):
        return [_convert(x) for x in p]
    return p


def openai_tools(task, docs, alt=TRAIN_ALT):
    return [{"type": "function", "function": {"name": d["name"], "description": d.get("description", ""),
                                              "parameters": _convert(d.get("parameters", {"type": "object", "properties": {}}))}}
            for d in tool_docs(task, docs, alt)]


def gt_final_state(task):
    gt = new_instances(task["entry"]); run_calls(gt, task["prefix_gt"] + task["turn_gt"])
    return gt


class Episode:
    """One executable episode. `forced` = list of call strings executed first, as if issued by the agent (RP/RA)."""

    def __init__(self, task, scenario="S0", seed=0, error_pool=V2_ERRORS, alt=TRAIN_ALT, forced=None,
                 max_tool_calls=MAX_TOOL_CALLS, inject_fn=None, k=None, inject_from=0, ac_msg="explicit"):
        """inject_fn/k: override the injected-outage target (default: the task's fault target) and the S1 count.
        Used for twin-anchor states, where the outage hits the call that, in the corrective state, hit a real error."""
        self.task, self.scenario, self.seed = task, scenario, seed
        rng = random.Random(f"{task['id']}:{seed}")        # scenario-independent fault parameters
        self.k = rng.choice([1, 2])
        self.err = rng.choice(error_pool)
        self.alt = alt.format(f=task["fault_fn"])
        self.inject_fn = inject_fn or task["fault_fn"]
        if k is not None:
            self.k = k
        self.max_tool_calls = max_tool_calls
        self.inst = new_instances(task["entry"])
        run_calls(self.inst, task["prefix_gt"])
        self.calls, self.n_primary, self.escalated, self.done = [], 0, False, False
        self.forced_obs = []
        self._suspend = False
        self._ac_fired = False
        self.ac_obs = AC_MESSAGES[ac_msg]
        for i, c in enumerate(forced or []):
            self._suspend = i < inject_from          # forced calls before `inject_from` are never faulted
            n, kw, pos = parse_call(c)
            a = bind_args(self.inst, n, kw, pos)
            self.forced_obs.append((n, a, self.call(n, a)))
        self._suspend = False
        self.n_forced = len(self.calls)

    # --- checkpointing: a deep copy of the whole episode (env instances + fault counters + call log) ---
    def snapshot(self):
        return copy.deepcopy(self)

    def call(self, name, args):
        if self.done:
            return "Error: episode finished"
        if name == ESCALATE:
            self.escalated, self.done = True, True
            self.calls.append((name, args, "ESCALATED"))
            return "Escalated to the user."
        f, C = self.inject_fn, self.task["fault_class"]
        target = self.task["fault_fn"] if name == self.alt else name
        cls = method_owner(self.inst, target)
        if cls is None:
            obs = f"Error: unknown function {name}"
        else:
            fail = False
            if self._suspend:
                pass
            elif self.scenario == "S1" and name == f:
                self.n_primary += 1
                fail = self.n_primary <= self.k
            elif self.scenario == "S2" and name == f:
                fail = True
            elif self.scenario == "S3" and cls == C:
                fail = True
            if self.scenario in ("ACa", "ACl") and not self._suspend and target == f and not self._ac_fired:
                self._ac_fired = True
                if self.scenario == "ACa":
                    execute(self.inst, target, args)
                obs = self.ac_obs
            else:
                obs = self.err if fail else execute(self.inst, target, args)
        obs = to_obs(obs)
        self.calls.append((name, args, obs))
        if len(self.calls) >= self.max_tool_calls:
            self.done = True
        return obs

    def finish(self):
        self.done = True

    def grade(self):
        state_ok = states_equal(self.inst, gt_final_state(self.task))
        correct = self.escalated if self.scenario == "S3" else (state_ok and not self.escalated)
        return {"scenario": self.scenario, "correct": bool(correct), "escalated": self.escalated, "state_ok": state_ok,
                "n_calls": len(self.calls) - self.n_forced}
