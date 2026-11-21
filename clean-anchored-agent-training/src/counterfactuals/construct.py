"""Outcome-verified corrective decision states, counterfactual branches, and clean anchors.

A *decision state* is (task, forced prefix of calls). It is never stored as a pickled env: it is reconstructed by
deterministic replay (`Episode(task, forced=...)`), and every branch is executed from `snapshot()` deep copies of that
same reconstructed episode, so the two actions of a pair are compared from a bit-identical state (asserted via
`state_hash`). No continuation is ever spliced from another state.

Corrective states (real BFCL errors, never rewritten):
  RP  the agent skipped ground-truth call G[j]; the first later call G[m] returns a real error.
      a+ = G[j] (do the precondition), then G[m], G[m+1:].
  RA  the agent issued G[i] with a one-character typo in one string argument; the call returns a real error and the
      correct value is visible in the context (request, history, or an earlier observation). a+ = G[i].
Negative candidates (CPU stage) = the outage responses: identical retry, alternative endpoint (if it exists for that
function), escalate. On GPU, negatives are the policy's own sampled actions (see scripts/construct_correction_pairs.py).

A pair (s, a+, a-) is RETAINED only if, under the same continuation policy and tool budget, the a+ branch passes the
state-based grader and the a- branch fails. Continuations are deterministic here, so one execution per branch is exact;
for stochastic (policy) continuations use `verify_stochastic`, which needs a positive lower confidence bound.

Clean anchors for each corrective state (the behaviour the correction must not overwrite):
  T-out   twin: same task, correct prefix, the SAME call returns an injected outage instead of a real error.
          Correct action: identical retry (verified to complete the task).
  T-clean twin: same task, correct prefix, the call succeeds. Correct action: the next ground-truth call.
Generic anchors (baseline): ground-truth steps of random fault-free train episodes.
"""
from __future__ import annotations

import json
import math
import random

from src.counterfactuals.policies import OutageRule, Repairer
from src.environment.bfcl_env import ESCALATE, TRAIN_ERRORS, Episode, obs_kind
from src.replay.replay import bind_args, call_to_str, parse_call, state_hash

SYSTEM = ("You are a helpful assistant that completes user requests by calling the provided tools. Tool calls can fail. "
          "If a call fails, you may retry it; if a function keeps failing, use its backup endpoint if one exists; if every "
          "way to complete the request keeps failing, call escalate_to_user instead of continuing. Do not give up on a "
          "request that can still be completed.")


class Counter:
    """Counts environment tool executions spent on data construction (reported as data-generation cost)."""
    def __init__(self):
        self.n = 0


def history_messages(task, system=SYSTEM):
    from src.replay.replay import new_instances, run_calls
    msgs = [{"role": "system", "content": system}]
    inst = new_instances(task["entry"])
    for h in task["history"]:
        msgs.append({"role": "user", "content": h["user"]})
        calls = []
        for c in h["calls"]:
            n, kw, pos = parse_call(c)
            calls.append((n, bind_args(inst, n, kw, pos)))
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]})
        for o in run_calls(inst, h["calls"]):
            msgs.append({"role": "tool", "content": o[:2000]})
        msgs.append({"role": "assistant", "content": "Done."})
    msgs.append({"role": "user", "content": task["user"]})
    return msgs


def call_msg(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


def episode_messages(task, ep, system=SYSTEM):
    msgs = history_messages(task, system)
    for n, a, o in ep.calls:
        msgs.append(call_msg(n, a))
        msgs.append({"role": "tool", "content": ("Escalated to the user." if o == "ESCALATED" else o)[:2000]})
    return msgs


def _bound(ep, call):
    n, kw, pos = parse_call(call)
    return n, bind_args(ep.inst, n, kw, pos)


def _native(o):
    return obs_kind(o) == "native"


def make_episode(task, forced, counter, **kw):
    ep = Episode(task, kw.pop("scenario", "S0"), kw.pop("seed", 0), error_pool=kw.pop("error_pool", TRAIN_ERRORS),
                 forced=forced, **kw)
    counter.n += len(forced)
    return ep


# ---------------- corrective decision states ----------------
def rp_states(task, counter):
    G, out = task["turn_gt"], []
    for j in range(len(G) - 1):
        ep = make_episode(task, [], counter)
        m_fail = None
        for m in range(len(G)):
            if m == j:
                continue
            n, a = _bound(ep, G[m]); o = ep.call(n, a); counter.n += 1
            if _native(o):
                m_fail = m
                break
        if m_fail is None or m_fail < j:
            continue
        forced = [G[i] for i in range(m_fail + 1) if i != j]
        repaired = [G[j]] + G[m_fail:]
        out.append({"kind": "RP", "task_id": task["id"], "forced": forced, "repair_plan": repaired, "skip": j, "m": m_fail,
                    "correct_prefix": G[:m_fail], "pivot_call": G[m_fail]})
    return out


def typo(v, rng):
    i = rng.randrange(len(v))
    return v[:i] + v[i + 1:]


def ra_states(task, counter, seed=0):
    G, out = task["turn_gt"], []
    rng = random.Random(f"{task['id']}:ra:{seed}")
    for i, c in enumerate(G):
        name, kw, pos = parse_call(c)
        ep0 = make_episode(task, [], counter)
        args = bind_args(ep0.inst, name, kw, pos)
        for k, v in args.items():
            if not isinstance(v, str) or len(v) < 3:
                continue
            bad = dict(args); bad[k] = typo(v, rng)
            bad_c = call_to_str(name, bad)
            ep = make_episode(task, G[:i] + [bad_c], counter)
            o = ep.calls[-1][2]
            if not _native(o) or any(_native(x[2]) for x in ep.calls[:-1]):
                continue
            ctx = task["user"] + json.dumps(task["history"], default=str) + json.dumps([x[2] for x in ep.calls[:-1]], default=str)
            if v not in ctx:          # the repair must be inferable from the context
                continue
            out.append({"kind": "RA", "task_id": task["id"], "forced": G[:i] + [bad_c], "repair_plan": G[i:], "call_index": i,
                        "arg": k, "bad_value": bad[k], "correct_prefix": G[:i], "pivot_call": c})
    return out


def candidate_actions(state, task, ep):
    """a+ = first repair action; negatives = outage responses at this state."""
    n_err, a_err, _ = ep.calls[-1]
    pos = _bound(ep, state["repair_plan"][0])
    negs = [("retry", (n_err, a_err)), ("escalate", (ESCALATE, {"reason": "The service appears to be unavailable."}))]
    if n_err == task["fault_fn"]:
        negs.append(("alternative", (ep.alt, a_err)))
    return pos, negs


def run_branch(snap, action, pending, policy, counter):
    ep = snap.snapshot()
    o = ep.call(*action); counter.n += 1
    before = len(ep.calls)
    policy.continue_from(ep, pending, last=(action[0], action[1], o))
    counter.n += len(ep.calls) - before
    return ep.grade()["correct"], ep


def verify_state(task, state, counter):
    """Builds the state, checks it is a genuine failure point, and branches a+ and every negative under both
    continuation policies. Returns the state record with per-branch outcomes and the retained pairs."""
    snap = make_episode(task, state["forced"], counter)
    h0 = state_hash(snap.inst)
    # validity: naive continuation that ignores the error must fail, the full repair must succeed
    naive = snap.snapshot()
    rest = state["repair_plan"][1:] if state["kind"] == "RP" else state["repair_plan"][1:]
    for c in (task["turn_gt"][state["m"] + 1:] if state["kind"] == "RP" else rest):
        naive.call(*_bound(naive, c)); counter.n += 1
    state["naive_fails"] = not naive.grade()["correct"]
    pos, negs = candidate_actions(state, task, snap)
    pending_pos = state["repair_plan"][1:]
    res = {}
    for pname, pol in (("surrogate", OutageRule()), ("repairer", Repairer(state["repair_plan"]))):
        ok_pos, _ = run_branch(snap, pos, pending_pos, pol, counter)
        r = {"pos": ok_pos}
        for nm, act in negs:
            ok, _ = run_branch(snap, act, pending_pos, pol, counter)
            r[nm] = ok
        res[pname] = r
    assert state_hash(snap.inst) == h0, "branching mutated the checkpoint"
    state["outcomes"] = res
    state["state_hash"] = h0
    state["pos_action"] = {"name": pos[0], "arguments": pos[1]}
    state["neg_actions"] = {nm: {"name": a[0], "arguments": a[1]} for nm, a in negs}
    state["messages"] = episode_messages(task, snap)
    state["error_obs"] = snap.calls[-1][2]
    sur = res["surrogate"]
    state["retained_pairs"] = [nm for nm, _ in negs if sur["pos"] and not sur[nm]] if state["naive_fails"] else []
    return state


# ---------------- clean anchors ----------------
def twin_anchors(task, state, counter):
    """T-out and T-clean twins of a corrective state, each verified by execution."""
    out = []
    prefix, pivot = state["correct_prefix"], state["pivot_call"]
    n, kw, pos = parse_call(pivot)
    # T-out: the pivot call gets an injected transient outage (k=1)
    ep = make_episode(task, prefix + [pivot], counter, scenario="S1", inject_fn=n, k=1, inject_from=len(prefix))
    if ep.calls and obs_kind(ep.calls[-1][2]) == "injected":
        snap = ep.snapshot()
        a = (ep.calls[-1][0], ep.calls[-1][1])
        ok, _ = run_branch(snap, a, task["turn_gt"][len(prefix) + 1:], OutageRule(), counter)
        if ok:
            negs = {"escalate": {"name": ESCALATE, "arguments": {"reason": "The service appears to be unavailable."}}}
            if a[0] == task["fault_fn"]:
                negs["alternative"] = {"name": snap.alt, "arguments": a[1]}
            out.append({"kind": "T-out", "task_id": task["id"], "of": state_key(state), "messages": episode_messages(task, snap),
                        "action": {"name": a[0], "arguments": a[1]}, "negs": negs, "state_hash": state_hash(snap.inst)})
    # T-clean: the pivot call succeeds; target = next ground-truth call, or the final reply if it was the last call
    ep = make_episode(task, prefix + [pivot], counter)
    if ep.calls and obs_kind(ep.calls[-1][2]) == "ok":
        nxt = task["turn_gt"][len(prefix) + 1:]
        if nxt:
            a = _bound(ep, nxt[0])
            ok, _ = run_branch(ep.snapshot(), a, nxt[1:], OutageRule(), counter)
            action = {"name": a[0], "arguments": a[1]}
        else:
            ok, action = ep.grade()["correct"], {"final_text": "Done."}
        if ok:
            last = ep.calls[-1]
            negs = {"escalate": {"name": ESCALATE, "arguments": {"reason": "The service appears to be unavailable."}},
                    "duplicate": {"name": last[0], "arguments": last[1]}}
            out.append({"kind": "T-clean", "task_id": task["id"], "of": state_key(state), "messages": episode_messages(task, ep),
                        "action": action, "negs": negs, "state_hash": state_hash(ep.inst)})
    return out


def generic_anchors(tasks, n, counter, seed=0, exclude_ids=()):
    rng = random.Random(seed)
    pool = [(t, i) for t in tasks if t["id"] not in exclude_ids for i in range(len(t["turn_gt"]) + 1)]
    rng.shuffle(pool)
    out = []
    for t, i in pool[:n]:
        ep = make_episode(t, t["turn_gt"][:i], counter)
        if i < len(t["turn_gt"]):
            a = _bound(ep, t["turn_gt"][i]); action = {"name": a[0], "arguments": a[1]}
        else:
            action = {"final_text": "Done."}
        out.append({"kind": "G-clean", "task_id": t["id"], "messages": episode_messages(t, ep), "action": action,
                    "state_hash": state_hash(ep.inst)})
    return out


def state_key(s):
    return f"{s['task_id']}|{s['kind']}|{s.get('skip', s.get('call_index'))}|{s.get('arg', '')}"


def verify_stochastic(succ_pos, succ_neg, z=1.64):
    """Policy-continuation verification: keep only if the success-rate difference has a positive lower bound
    (normal approximation with per-arm variance; requires repeated paired executions from the same checkpoint)."""
    k1, n1 = sum(succ_pos), len(succ_pos)
    k2, n2 = sum(succ_neg), len(succ_neg)
    if n1 == 0 or n2 == 0:
        return False, 0.0
    p1, p2 = (k1 + 0.5) / (n1 + 1), (k2 + 0.5) / (n2 + 1)
    d = k1 / n1 - k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return d - z * se > 0, d
