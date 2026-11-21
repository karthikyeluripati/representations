"""Feasibility of an 'ambiguous commit' fault suite on the existing BFCL tasks (CPU only, no model).

Fault: the task's fault target f returns
    "Error: request timed out after the server received it; the operation may or may not have been applied."
  AC_applied : f WAS executed (state changed)  -> correct = verify, do NOT re-issue f, finish the turn
  AC_lost    : f was NOT executed              -> correct = verify, re-issue f once, finish the turn
Identical observation in both variants, so the only policy that passes both is verify-then-decide.

A task is diagnostic iff
  (1) non-idempotent: executing f twice gives a final state != ground truth (blind retry fails AC_applied), and
  (2) verifiable: some read-only function of f's class, with arguments available in context (no required params, or
      required params all shared with f's own arguments), returns different output with vs without f applied.
(AC_lost is automatically diagnostic: never retrying leaves f's effect missing, since f changes state by construction.)
"""
import collections
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); sys.path.insert(0, PROJ)
from src.environment.bfcl_env import AC_MESSAGES, ESCALATE, Episode, load_bfcl  # noqa: E402
from src.replay.replay import bind_args, execute, new_instances, parse_call, public_state, run_calls, states_equal  # noqa: E402

READ_PREFIX = ("get", "list", "display", "view", "check", "ls", "cat", "find", "show", "search", "pwd", "du", "wc", "tail",
               "head", "grep", "sort", "diff", "estimate", "retrieve", "fetch", "read", "verify", "_get", "query", "lookup")


def fault_index(t):
    return next(i for i, c in enumerate(t["turn_gt"]) if parse_call(c)[0] == t["fault_fn"])


def state_at(t, upto_extra):
    inst = new_instances(t["entry"]); run_calls(inst, t["prefix_gt"] + upto_extra); return inst


def main():
    _, _, docs = load_bfcl()
    tasks = [t for t in json.load(open(os.path.join(PROJ, "data", "tasks_v1.json"), encoding="utf-8")) if t["usable"]]
    rows, why = [], collections.Counter()
    for t in tasks:
        i = fault_index(t); gt = t["turn_gt"]; f = gt[i]
        final = state_at(t, gt)
        dup = state_at(t, gt[:i + 1] + [f] + gt[i + 1:])
        non_idem = not states_equal(dup, final)
        # verifiers: read-only functions of the fault class, evaluated right after f vs right before f
        n, kw, pos = parse_call(f)
        before, after = state_at(t, gt[:i]), state_at(t, gt[:i + 1])
        fargs = bind_args(before, n, kw, pos)
        ver, dec = [], []
        for d in docs.get(t["fault_class"], []):
            name = d["name"]
            if name == n or not name.lower().lstrip("_").startswith(READ_PREFIX):
                continue
            req = d.get("parameters", {}).get("required", [])
            if any(r not in fargs for r in req):
                continue
            args = {r: fargs[r] for r in req}
            b2, a2 = copy.deepcopy(before), copy.deepcopy(after)
            ob, oa = execute(b2, name, args), execute(a2, name, args)
            ro = public_state(b2[t["fault_class"]]) == public_state(before[t["fault_class"]])
            if ro and str(ob) != str(oa) and not str(oa).startswith("Error"):
                ver.append(name)
                # decisive: some argument value of f is visible in the post-call output but absent before it, so the
                # outcome can be read from this one observation without knowing the earlier state
                vals = [json.dumps(v) if not isinstance(v, str) else v for v in fargs.values() if v not in (None, "", [], {})]
                vals = [v.strip('"') for v in vals if len(v.strip('"')) >= 3]
                if any(v in str(oa) and v not in str(ob) for v in vals):
                    dec.append(name)
        diag = non_idem and bool(ver)
        why[("non_idempotent" if non_idem else "idempotent", "verifiable" if ver else "no_verifier")] += 1
        rows.append({"task_id": t["id"], "split": t["split"], "fault_fn": n, "fault_class": t["fault_class"],
                     "non_idempotent": non_idem, "verifiers": ver, "diagnostic": diag,
                     "decisive_verifiers": dec, "decisive": diag and bool(dec)})
    print("all usable tasks:", len(rows), dict(why))
    for sp in ("train", "dev", "test", "heldout"):
        rs = [r for r in rows if r["split"] == sp]
        print(f"  {sp:8s} n={len(rs):3d} diagnostic={sum(r['diagnostic'] for r in rs):3d}  non_idem={sum(r['non_idempotent'] for r in rs):3d}"
              f"  by class={dict(collections.Counter(r['fault_class'] for r in rs if r['diagnostic']))}")
        print(f"           decisive={sum(r['decisive'] for r in rs):3d}  by class={dict(collections.Counter(r['fault_class'] for r in rs if r['decisive']))}")
    print("diagnostic fault functions:", collections.Counter(r["fault_fn"] for r in rows if r["diagnostic"]).most_common(15))
    print("verifiers used:", collections.Counter(v for r in rows for v in r["verifiers"]).most_common(12))
    out = os.path.join(PROJ, "results", "feasibility_ambiguous_commit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("wrote", out)


def run_policy(t, scenario, policy, verifier, ac_msg="explicit"):
    """Scripted policies over the ground-truth turn. verify: read the verifier before f; on the ambiguous error, read it
    again and re-issue f only if the output did not change. blind_retry / never_retry / escalate as named."""
    ep = Episode(t, scenario=scenario, seed=0, ac_msg=ac_msg)
    i = fault_index(t)
    for j, c in enumerate(t["turn_gt"]):
        n, kw, pos = parse_call(c); a = bind_args(ep.inst, n, kw, pos)
        if j == i and policy == "verify":
            vn, va = verifier
            pre = ep.call(vn, va)
        o = ep.call(n, a)
        if o in AC_MESSAGES.values():
            if policy == "blind_retry":
                ep.call(n, a)
            elif policy == "escalate":
                ep.call(ESCALATE, {"reason": "operation status unknown"}); break
            elif policy == "verify":
                if ep.call(vn, va) == pre:
                    ep.call(n, a)
    return ep.grade()["correct"]


def validate(tasks, rows):
    by = {t["id"]: t for t in tasks}
    res = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if not r["diagnostic"]:
            continue
        t = by[r["task_id"]]; i = fault_index(t)
        n, kw, pos = parse_call(t["turn_gt"][i])
        st = state_at(t, t["turn_gt"][:i]); fa = bind_args(st, n, kw, pos)
        d = next(d for d in load_bfcl()[2][t["fault_class"]] if d["name"] == r["verifiers"][0])
        ver = (d["name"], {q: fa[q] for q in d.get("parameters", {}).get("required", [])})
        for msg in AC_MESSAGES:
            for sc in ("ACa", "ACl", "S0"):
                if sc == "S0" and msg != "explicit":
                    continue
                for pol in ("verify", "blind_retry", "never_retry", "escalate"):
                    ok = run_policy(t, sc, pol, ver, msg)
                    res[(sc if sc == "S0" or msg == "explicit" else f"{sc}_{msg}", pol)][0] += ok
                    res[(sc if sc == "S0" or msg == "explicit" else f"{sc}_{msg}", pol)][1] += 1
    print("scripted validation on diagnostic tasks (correct / n):")
    for pol in ("verify", "blind_retry", "never_retry", "escalate"):
        print(f"  {pol:12s}", "  ".join(f"{sc}: {res[(sc, pol)][0]}/{res[(sc, pol)][1]}" for sc in
                                        ("ACa", "ACl", "S0", "ACa_para", "ACl_para", "ACa_implicit", "ACl_implicit")))


if __name__ == "__main__":
    main()
    tasks = [t for t in json.load(open(os.path.join(PROJ, "data", "tasks_v1.json"), encoding="utf-8")) if t["usable"]]
    validate(tasks, json.load(open(os.path.join(PROJ, "results", "feasibility_ambiguous_commit.json"), encoding="utf-8")))
