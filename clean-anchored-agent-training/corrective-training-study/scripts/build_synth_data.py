"""Synthetic-API training decisions (docs/method_synthetic.md). Tokenizer-free; every decision verified by execution.

32 synthetic classes x 2 tasks per kind. Under the IAR convention:
  create (non-idempotent) + timeout-type TRAIN error -> check (list) ; after the check: finish (applied) / re-issue (lost)
  create + "503" rejection                            -> retry directly
  set / remove (idempotent) + any TRAIN error          -> retry directly (safe whether or not it applied)
The OLD-convention control answers create + timeout with a direct retry; everything else identical; cycled to the same size.
Output: data/synth_v1/{syn_iar.jsonl, syn_old.jsonl, MANIFEST.json}
"""
import argparse
import collections
import hashlib
import itertools
import json
import os

import _common
from src.counterfactuals.construct import SYSTEM
from src.environment.bfcl_env import TRAIN_ERRORS
from verification.synthetic_apis import DOMAINS, SynthClass, make_tasks, run

TIMEOUTS = TRAIN_ERRORS[1:]; REJECT = TRAIN_ERRORS[0]
FINISH = {"final_text": "Done."}


def call_msg(a):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": a["name"], "arguments": a["arguments"]}}]}


def obs_after(cls, task, applied, extra):
    st = json.loads(json.dumps(task["state"]))
    st["records"] = {int(k): v for k, v in st["records"].items()}
    if applied:
        cls.execute(st, task["call"]["name"], task["call"]["arguments"])
    return cls.execute(st, extra["name"], extra["arguments"])


def rec(kind, cls, task, msgs, chosen, sid):
    return {"kind": kind, "task_id": f"synth:{cls.domain}", "tools": cls.tools(), "msgs": msgs, "chosen": chosen, "state": sid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-classes", type=int, default=32); ap.add_argument("--tasks-per-kind", type=int, default=2)
    ap.add_argument("--version", default="synth_v1")
    a = ap.parse_args()
    assert a.n_classes <= len(DOMAINS)
    OUT = os.path.join(_common.STUDY, "data", a.version)
    iar, old, drops = [], [], collections.Counter()
    for ci in range(a.n_classes):
        cls = SynthClass(ci)
        lst = {"name": cls.fn["list"], "arguments": {}}
        for ti, t in enumerate(make_tasks(cls, a.tasks_per_kind)):
            base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t["user"]}, call_msg(t["call"])]
            sid = f"{cls.domain}|{t['kind']}|{ti}"
            if t["kind"] == "create":
                for e in TIMEOUTS:
                    m = base + [{"role": "tool", "content": e}]
                    # verify: check-then-decide succeeds in both twins; blind retry fails when applied; skip fails when lost
                    ok = (run(cls, t, True, [lst, FINISH]) and run(cls, t, False, [lst, t["call"]])
                          and not run(cls, t, True, [t["call"]]) and not run(cls, t, False, [FINISH]))
                    if not ok:
                        drops["create:unverified"] += 1; continue
                    iar.append(rec("syn_check", cls, t, m, lst, sid + f"|{e[:12]}|err"))
                    for applied in (True, False):
                        o = obs_after(cls, t, applied, lst)
                        assert (t["value"] in o) == applied, "check must be decisive"
                        m2 = m + [call_msg(lst), {"role": "tool", "content": o}]
                        iar.append(rec("syn_post", cls, t, m2, FINISH if applied else t["call"], sid + f"|{e[:12]}|post|{applied}"))
                    old.append(rec("syn_old_retry", cls, t, m, t["call"], sid + f"|{e[:12]}|old"))
                assert run(cls, t, False, [t["call"]])
                r = rec("syn_reject_retry", cls, t, base + [{"role": "tool", "content": REJECT}], t["call"], sid + "|rej")
                iar.append(r); old.append(r)
            else:
                for e in TRAIN_ERRORS:
                    if not (run(cls, t, True, [t["call"]]) and run(cls, t, False, [t["call"]])):
                        drops["idem:unverified"] += 1; continue
                    r = rec("syn_idem_retry", cls, t, base + [{"role": "tool", "content": e}], t["call"], sid + f"|{e[:12]}|idem")
                    iar.append(r); old.append(r)
    old_c = list(itertools.islice(itertools.cycle(old), len(iar)))
    os.makedirs(OUT, exist_ok=True)
    man = {"version": a.version, "n_classes": a.n_classes, "tasks_per_kind": a.tasks_per_kind, "n": len(iar), "counts_iar": dict(collections.Counter(r["kind"] for r in iar)),
           "counts_old_before_cycling": dict(collections.Counter(r["kind"] for r in old)), "dropped": dict(drops)}
    for name, rows in (("syn_iar.jsonl", iar), ("syn_old.jsonl", old_c)):
        p = os.path.join(OUT, name)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
