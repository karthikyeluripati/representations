"""Check-all-writes (CAW) training data (docs/method_caw.md). Tokenizer-free; every decision verified by execution.

Rule taught: after a timeout-type failure of ANY state-changing call (create, set, remove/cancel), CHECK the state
and then decide (applied -> continue/finish; lost -> re-issue). Clean rejections ("503") and READ calls -> retry
directly. No idempotency inference is needed, so no verb-based shortcut is available.
Synthetic part: 64 classes x (create, set, remove: 1 timeout state + 2 post-check states each; 503 retry for each;
1 read retry) = 13 records/class = 832, the same size as SYN64. BFCL part: method_iar.md decisions without the
idempotent-retry records (which contradict CAW), cycled to 517 so totals match SYN64 exactly.
Output: data/caw_v1/{caw.jsonl, MANIFEST.json}
"""
import collections
import hashlib
import itertools
import json
import os

import _common
from src.counterfactuals.construct import SYSTEM
from src.environment.bfcl_env import TRAIN_ERRORS
from verification.synthetic_apis import SynthClass, make_read_task, make_tasks, run

OUT = os.path.join(_common.STUDY, "data", "caw_v1")
TIMEOUTS = TRAIN_ERRORS[1:]; REJECT = TRAIN_ERRORS[0]; FINISH = {"final_text": "Done."}
N_CLASSES = 64


def call_msg(a):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": a["name"], "arguments": a["arguments"]}}]}


def obs_after(cls, task, applied, extra):
    st = json.loads(json.dumps(task["state"])); st["records"] = {int(k): v for k, v in st["records"].items()}
    if applied:
        cls.execute(st, task["call"]["name"], task["call"]["arguments"])
    return cls.execute(st, extra["name"], extra["arguments"])


def rec(kind, cls, msgs, chosen, sid):
    return {"kind": kind, "task_id": f"synth:{cls.domain}", "tools": cls.tools(), "msgs": msgs, "chosen": chosen, "state": sid}


def main():
    syn, drops = [], collections.Counter()
    for ci in range(N_CLASSES):
        cls = SynthClass(ci); lst = {"name": cls.fn["list"], "arguments": {}}
        e = TIMEOUTS[ci % 2]                                     # one timeout string per class, alternating
        for t in make_tasks(cls, 1):
            base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t["user"]}, call_msg(t["call"])]
            sid = f"{cls.domain}|{t['kind']}"
            m = base + [{"role": "tool", "content": e}]
            ok = run(cls, t, True, [lst, FINISH]) and run(cls, t, False, [lst, t["call"]]) and not run(cls, t, False, [FINISH])
            if not ok:
                drops[f"{t['kind']}:unverified"] += 1; continue
            post = {}
            for applied in (True, False):
                post[applied] = obs_after(cls, t, applied, lst)
            if post[True] == post[False]:
                drops[f"{t['kind']}:indecisive"] += 1; continue
            syn.append(rec("caw_check", cls, m, lst, sid + "|err"))
            for applied in (True, False):
                m2 = m + [call_msg(lst), {"role": "tool", "content": post[applied]}]
                syn.append(rec("caw_post", cls, m2, FINISH if applied else t["call"], sid + f"|post|{applied}"))
            assert run(cls, t, False, [t["call"]])
            syn.append(rec("caw_reject_retry", cls, base + [{"role": "tool", "content": REJECT}], t["call"], sid + "|rej"))
        rt = make_read_task(cls)
        base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rt["user"]}, call_msg(rt["call"])]
        syn.append(rec("caw_read_retry", cls, base + [{"role": "tool", "content": e}], rt["call"], f"{cls.domain}|read"))
    bfcl = [json.loads(l) for l in open(os.path.join(_common.STUDY, "data", "iar_v1", "iar.jsonl"), encoding="utf-8")]
    bfcl = [r for r in bfcl if r["kind"] != "idem_retry"]
    bfcl_c = list(itertools.islice(itertools.cycle(bfcl), 517))
    rows = bfcl_c + syn
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "caw.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"version": "caw_v1", "n_classes": N_CLASSES, "n_synthetic": len(syn), "n_bfcl": len(bfcl_c), "n_bfcl_unique": len(bfcl),
           "n": len(rows), "counts": dict(collections.Counter(r["kind"] for r in rows)), "dropped": dict(drops),
           "caw.jsonl": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
