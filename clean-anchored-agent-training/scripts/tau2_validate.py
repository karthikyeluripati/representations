"""Scripted validation of tau2 airline uncertain-outcome episodes (CPU). Expected separation, as on BFCL:
verify-then-decide passes ACa and ACl; blind retry fails ACa; never-retry fails ACl; on idempotent control writes
blind retry passes both. Usage: python scripts/tau2_validate.py"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from src.environment.bfcl_env import AC_MESSAGES  # noqa: E402
from src.environment.tau2_airline import Tau2Episode, _load, events  # noqa: E402


def reader(ep, w):
    a = dict(w.arguments)
    if "reservation_id" in a:
        return "get_reservation_details", {"reservation_id": a["reservation_id"]}
    uid = next((x[1].get("user_id") for x in ep.calls if "user_id" in x[1]), None) or a.get("user_id")
    return "get_user_details", {"user_id": uid}


def play(tid, i, sc, msg, policy):
    ep = Tau2Episode(tid, i, sc, AC_MESSAGES[msg])
    acts = ep.task.evaluation_criteria.actions; w = acts[i]
    plan = []
    if policy == "verify":
        rn, ra = reader(ep, w); ep.call(rn, ra)
        plan = ([] if sc == "ACa" else [w]) + list(acts[i + 1:])
    elif policy == "blind_retry":
        plan = [w] + list(acts[i + 1:])
    elif policy == "never_retry":
        plan = list(acts[i + 1:])
    for a in plan:
        ep.call(a.name, dict(a.arguments))
    return ep.grade()["correct"]


def main():
    _load(); E = events()
    res = collections.defaultdict(lambda: [0, 0])
    for kind in ("diagnostic", "control"):
        for tid, i, fn in E[kind]:
            for msg in ("explicit", "para", "implicit"):
                for sc in ("ACa", "ACl"):
                    for pol in ("verify", "blind_retry", "never_retry"):
                        ok = play(tid, i, sc, msg, pol)
                        res[(kind, sc, pol)][0] += ok; res[(kind, sc, pol)][1] += 1
    for kind in ("diagnostic", "control"):
        print(kind, len(E[kind]), "events")
        for pol in ("verify", "blind_retry", "never_retry"):
            print(f"  {pol:12s}", "  ".join(f"{sc}: {res[(kind, sc, pol)][0]}/{res[(kind, sc, pol)][1]}" for sc in ("ACa", "ACl")))


if __name__ == "__main__":
    main()
