"""Scripted continuation policies used to verify branch outcomes on CPU.

On GPU the continuation policy is the trained policy itself (src/evaluation/agent_loop.py, CSO-style). Without a
checkpoint, two scripted policies bracket the possibilities and are reported separately:

  * `OutageRule` (surrogate of the over-generalising recovery-SFT policy): replays the planned calls; on ANY error
    it applies the outage procedure (identical retry up to 3 attempts, then the alternative endpoint twice if one
    exists for that function, then escalate). The previous project measured 92-93% agreement of its SFT policy with
    this rule and a 38% rate of applying it to real held-out errors.
  * `Repairer` (competent oracle): like OutageRule for injected outages, but on a real error it re-plans from the
    ground truth (performs the missing precondition / the correct argument) before continuing.

A branch outcome is graded by the episode's own state-based grader, with the same tool-call budget for both branches.
"""
from __future__ import annotations

from src.environment.bfcl_env import ESCALATE, obs_kind
from src.replay.replay import bind_args, parse_call


def _args(ep, call):
    n, kw, pos = parse_call(call)
    return n, bind_args(ep.inst, n, kw, pos)


def _is_err(obs, native_as_outage):
    k = obs_kind(obs)
    return k == "injected" or (native_as_outage and k == "native")


def handle_outage(ep, name, args, attempts_done=1):
    """Outage procedure after `attempts_done` failed attempts of (name, args). Returns True if the call succeeded."""
    for _ in range(3 - attempts_done):
        if ep.done:
            return False
        if not _is_err(ep.call(name, args), True):
            return True
    if name == ep.task["fault_fn"]:
        for _ in range(2):
            if ep.done:
                return False
            if not _is_err(ep.call(ep.alt, args), True):
                return True
    if not ep.done:
        ep.call(ESCALATE, {"reason": "service unavailable"})
    return False


class OutageRule:
    name = "outage_rule_surrogate"

    def continue_from(self, ep, pending, last=None):
        """`last` = (name, args, obs) of the branch action just executed (its error, if any, is handled first)."""
        if last is not None and _is_err(last[2], True) and last[0] != ESCALATE:
            if not handle_outage(ep, last[0], last[1], attempts_done=1):
                return
        for c in pending:
            if ep.done:
                return
            n, a = _args(ep, c)
            if _is_err(ep.call(n, a), True) and not handle_outage(ep, n, a, attempts_done=1):
                return


class Repairer:
    """Competent continuation: injected outages -> outage procedure; real errors -> skip ahead to the repaired plan
    supplied by the caller (`repair_plan`)."""
    name = "repairer_oracle"

    def __init__(self, repair_plan):
        self.repair_plan = repair_plan

    def continue_from(self, ep, pending, last=None):
        if last is not None and last[0] != ESCALATE and obs_kind(last[2]) == "native":
            pending = self.repair_plan
        elif last is not None and obs_kind(last[2]) == "injected":
            if not handle_outage(ep, last[0], last[1], attempts_done=1):
                return
        for c in pending:
            if ep.done:
                return
            n, a = _args(ep, c)
            o = ep.call(n, a)
            if obs_kind(o) == "injected" and not handle_outage(ep, n, a, attempts_done=1):
                return


def reference_policy(ep):
    """Executable reference used for solvability checks: ground-truth calls + outage procedure; escalate in S3."""
    OutageRule().continue_from(ep, ep.task["turn_gt"])
