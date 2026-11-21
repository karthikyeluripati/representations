"""Outcome-twin pairs for the OTV method study (docs/method_outcome_twins.md).

ACa / ACl twins share the identical history up to and including the ambiguous-commit error on the task's fault
target f; only the hidden state differs. Every pair below is VERIFIED BY EXECUTION: the chosen action, followed by
the scripted correct continuation, succeeds under the BFCL state grader; the rejected action, followed by its natural
continuation, fails (in the twin(s) the pair is used for).

  (a) error-state pairs   state = ...f -> AC error.          chosen = verifier call
                          rejected in {retry f, skip f (continue / finish), escalate}; each fails in >= 1 twin
  (b) post-check pairs    state = (a) + verifier call + its observation (differs between twins)
                          ACa: chosen = continue (next gt call, or finish), rejected = re-issue f
                          ACl: chosen = re-issue f,                         rejected = continue
  (s1) outage anchors     state = ...f -> unambiguous injected outage (S1, k=1). chosen = retry f, rejected = escalate
Pairs are abstract ({"msgs", "chosen", "rejected"} with action dicts) and are rendered for TRL on the GPU host with
`verification.pairs.dpo_records` exactly like Baseline B.
"""
from __future__ import annotations

import json

from src.counterfactuals.construct import SYSTEM, episode_messages
from src.environment.bfcl_env import ESCALATE, TRAIN_ERRORS, Episode
from src.replay.replay import bind_args, parse_call

FINISH = {"final_text": "Done."}


def fault_index(task):
    return next(i for i, c in enumerate(task["turn_gt"]) if parse_call(c)[0] == task["fault_fn"])


def _bound(ep, call):
    n, kw, pos = parse_call(call)
    return {"name": n, "arguments": bind_args(ep.inst, n, kw, pos)}


def _do(ep, act):
    if "final_text" in act:
        ep.finish()
        return None
    return ep.call(act["name"], act["arguments"])


def state_before_fault(task, scenario, ac_msg="explicit", k=None):
    """Episode in which the agent has issued the turn's ground-truth calls before f, then f (which faults)."""
    ep = Episode(task, scenario=scenario, seed=0, ac_msg=ac_msg, error_pool=TRAIN_ERRORS, k=k)
    i = fault_index(task)
    for c in task["turn_gt"][:i]:
        a = _bound(ep, c); _do(ep, a)
    f = _bound(ep, task["turn_gt"][i])
    obs = _do(ep, f)
    return ep, f, obs


def rest_actions(ep, task):
    i = fault_index(task)
    return [_bound(ep, c) for c in task["turn_gt"][i + 1:]]


def run_plan(ep, plan_fn):
    """plan_fn(ep) -> list of actions, evaluated lazily so arguments bind against the live state."""
    e = ep.snapshot()
    for a in plan_fn(e):
        if e.done:
            break
        _do(e, a)
    return e.grade()["correct"]


def _vals(f):
    out = []
    for v in f["arguments"].values():
        v = v if isinstance(v, str) else json.dumps(v)
        if len(v) >= 3:
            out.append(v)
    return out


def decisive_value(f, obs_applied, obs_lost):
    """An argument value of f visible after f was applied and absent otherwise (the decisiveness criterion)."""
    return next((v for v in _vals(f) if v in str(obs_applied) and v not in str(obs_lost)), None)


def with_evidence(act, text):
    """Prefix an action with a one-sentence rationale (assistant content before the tool call, or final text)."""
    if "final_text" in act:
        return {"final_text": f"{text} {act['final_text']}"}
    return {**act, "content": text}


def evidence_text(fn, value, applied):
    if applied:
        return f"Evidence: the result shows {value!r}, so {fn} already took effect; I will not repeat it."
    return f"Evidence: the result does not show {value!r}, so {fn} did not take effect; I will issue it again."


CHECK_TEXT = "The outcome of {fn} is uncertain, so I will check the current state before deciding."


def build_task_pairs(task, verifier, ac_msg, evidence=False):
    """verifier = (name, args). Returns (pairs, report). Raises nothing: unverifiable candidates are dropped and counted."""
    pairs, rep = [], {"dropped": []}
    twins = {v: state_before_fault(task, v, ac_msg) for v in ("ACa", "ACl")}
    va = {"name": verifier[0], "arguments": verifier[1]}
    va_chosen = with_evidence(va, CHECK_TEXT.format(fn=task["fault_fn"])) if evidence else va
    msgs = {v: episode_messages(task, ep, SYSTEM) for v, (ep, _, _) in twins.items()}
    assert msgs["ACa"] == msgs["ACl"], "twins must share the identical observable history"
    f = twins["ACa"][1]

    def nxt(ep):
        r = rest_actions(ep, task)
        return r[0] if r else FINISH

    # chosen at the error state: verify, then the twin-correct decision, then the rest
    plans = {"verify": lambda v: (lambda e: [va] + ([f] if v == "ACl" else []) + rest_actions(e, task)),
             "retry": lambda v: (lambda e: [f] + rest_actions(e, task)),
             "skip": lambda v: (lambda e: rest_actions(e, task) or [FINISH]),
             "escalate": lambda v: (lambda e: [{"name": ESCALATE, "arguments": {"reason": "operation status unknown"}}])}
    ok = {(name, v): run_plan(twins[v][0], p(v)) for name, p in plans.items() for v in ("ACa", "ACl")}
    rep["outcomes"] = {f"{n}|{v}": o for (n, v), o in ok.items()}
    first = {"retry": f, "skip": nxt(twins["ACa"][0]), "escalate": {"name": ESCALATE, "arguments": {"reason": "operation status unknown"}}}
    if ok[("verify", "ACa")] and ok[("verify", "ACl")]:
        for neg in ("retry", "skip", "escalate"):
            if not (ok[(neg, "ACa")] and ok[(neg, "ACl")]):
                pairs.append({"type": "otv_a", "task_id": task["id"], "msgs": msgs["ACa"], "chosen": va_chosen, "rejected": first[neg],
                              "neg": neg, "state": f"{task['id']}|AC|{ac_msg}|err", "ac_msg": ac_msg})
            else:
                rep["dropped"].append(f"a:{neg}:not_wrong")
    else:
        rep["dropped"].append("a:verify_fails")
    # post-check pairs, one per twin
    post = {}
    for v, (ep, _, _) in twins.items():
        e = ep.snapshot(); post[v] = (e, _do(e, va))
    dval = decisive_value(f, post["ACa"][1], post["ACl"][1])
    if evidence and dval is None:
        rep["dropped"].append("b:no_decisive_value"); return pairs, rep
    for v, (e, vobs) in post.items():
        if vobs is None or str(vobs).startswith("Error"):
            rep["dropped"].append(f"b:{v}:verifier_error"); continue
        m = episode_messages(task, e, SYSTEM)
        cont = nxt(e)
        good, bad = (cont, f) if v == "ACa" else (f, cont)
        if evidence:                       # chosen states the TRUE evidence; rejected the false claim + its action
            good = with_evidence(good, evidence_text(f["name"], dval, v == "ACa"))
            bad = with_evidence(bad, evidence_text(f["name"], dval, v != "ACa"))
        is_f = lambda act: "final_text" not in act and act["name"] == f["name"] and act["arguments"] == f["arguments"]
        g = run_plan(e, lambda x, good=good: [good] + ([] if "final_text" in good else
                                                     (rest_actions(x, task) if is_f(good) else rest_actions(x, task)[1:])))
        b = run_plan(e, lambda x, bad=bad: [bad] + ([] if "final_text" in bad else
                                                   (rest_actions(x, task) if is_f(bad) else rest_actions(x, task)[1:])))
        if g and not b:
            pairs.append({"type": "otv_b", "task_id": task["id"], "msgs": m, "chosen": good, "rejected": bad, "neg": f"post_{v}",
                          "state": f"{task['id']}|{v}|{ac_msg}|post", "ac_msg": ac_msg, "twin": v, "evidence_value": dval})
        else:
            rep["dropped"].append(f"b:{v}:chosen_ok={g}:rejected_ok={b}")
    return pairs, rep


def build_s1_anchor(task):
    ep, f, obs = state_before_fault(task, "S1", k=1)
    if obs not in TRAIN_ERRORS:
        return None
    esc = {"name": ESCALATE, "arguments": {"reason": "service unavailable"}}
    if run_plan(ep, lambda e: [f] + rest_actions(e, task)) and not run_plan(ep, lambda e: [esc]):
        return {"type": "otv_s1", "task_id": task["id"], "msgs": episode_messages(task, ep, SYSTEM), "chosen": f, "rejected": esc,
                "neg": "escalate", "state": f"{task['id']}|S1|anchor"}
    return None
