"""Derive MCP-style tool annotations (readOnlyHint, idempotentHint) for every BFCL function by EXECUTION.

For every ground-truth call c in every usable task (prefix and turn), from the state just before c:
  read-only     executing c leaves every public state unchanged
  idempotent    executing c twice gives the same state as executing it once
A function is labelled readOnlyHint=True only if every observed call is read-only, and idempotentHint=False if ANY
observed call is non-idempotent (conservative). Functions never observed get no annotation.
Output: results/tool_annotations.json   {name: {"readOnlyHint": bool, "idempotentHint": bool, "n": int}}
"""
import collections
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); sys.path.insert(0, PROJ)
from src.replay.replay import execute_str, new_instances, state_hash  # noqa: E402


def snap(inst):
    return state_hash(inst)                  # the same canonical state hash used by the grader


def main():
    tasks = [t for t in json.load(open(os.path.join(PROJ, "data", "tasks_v1.json"), encoding="utf-8")) if t["usable"]]
    obs = collections.defaultdict(lambda: {"ro": True, "idem": True, "n": 0})
    for t in tasks:
        calls = t["prefix_gt"] + t["turn_gt"]
        inst = new_instances(t["entry"])
        for c in calls:
            name = c.split("(", 1)[0].strip()
            before = snap(inst)
            once = copy.deepcopy(inst); execute_str(once, c); s1 = snap(once)
            twice = copy.deepcopy(once); execute_str(twice, c); s2 = snap(twice)
            o = obs[name]; o["n"] += 1
            o["ro"] &= (s1 == before)
            o["idem"] &= (s2 == s1)
            inst = once
    out = {n: {"readOnlyHint": o["ro"], "idempotentHint": o["idem"], "n": o["n"]} for n, o in sorted(obs.items())}
    p = os.path.join(PROJ, "results", "tool_annotations.json")
    json.dump(out, open(p, "w", encoding="utf-8", newline="\n"), indent=1)
    c = collections.Counter(("read-only" if v["readOnlyHint"] else "idempotent write" if v["idempotentHint"] else "NON-idempotent write")
                            for v in out.values())
    print(len(out), "functions labelled:", dict(c))
    print("non-idempotent:", [n for n, v in out.items() if not v["idempotentHint"]])


if __name__ == "__main__":
    main()
