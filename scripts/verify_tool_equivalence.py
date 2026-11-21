"""Verify behavioral equivalence by execution, not text: every response is parsed, executed call by call in
ToolSchemaSpace's SimEnv (hard control), and its executed native-action multiset is compared with the
intended label.

Usage: python scripts/verify_tool_equivalence.py --tss PATH --data results/interface_invariance/benchmark.jsonl
Exits 1 on any violation.
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import iface_lib as L


class Executor:
    """Per-prompt executor: maps the model-visible catalog back to native actions."""

    def __init__(self, kind, native_tools, extra=None):
        from toolschema.operators import compile_spec
        from toolschema.simenv import SimEnv
        self.SimEnv = SimEnv
        self.kind, self.extra = kind, extra or {}
        self.native_names = {t["name"] for t in native_tools}
        self.native_cmap = compile_spec(native_tools, {"ops": []}).call_map
        if kind == "tss":
            self.cmap = self.extra["compiled"].call_map

    def execute(self, calls):
        """Returns (executed native actions, n_rejected, flags)."""
        if self.kind == "tss":
            env = self.SimEnv(self.cmap)
            for n, a in calls:
                env.call(n, a)
            flags = env.finish()
            return list(env.native_actions), env.n_rejected, list(flags or [])
        # alias layer: decode to native, then execute natively in SimEnv
        native, flags = L.decode_alias(calls, self.native_names, self.extra["amap"])
        env = self.SimEnv(self.native_cmap)
        for n, a in native:
            env.call(n, a)
        return list(env.native_actions), env.n_rejected, flags + list(env.finish() or [])


def verify_record(rec, executors):
    """Check every response of one task record. Returns list of error strings (empty = verified)."""
    errs, states = [], {}
    for r in rec["responses"]:
        calls, bad = L.parse(r["text"])
        ex = executors[r["prompt_key"]]
        native, rej, flags = ex.execute(calls)
        st, sq = L.h(L.state_of(native)), L.h(L.seq_of(native))
        if bad or rej or flags:
            errs.append(f"{rec['task_id']}/{r['rid']}: parse={bad} rejected={rej} flags={flags}")
        if st != r["final_state_hash"] or sq != r["seq_hash"]:
            errs.append(f"{rec['task_id']}/{r['rid']}: executed state differs from recorded")
        states.setdefault(r["behavior"], set()).add(st)
    g = states.get("G", set())
    if len(g) != 1:
        errs.append(f"{rec['task_id']}: equivalent responses of G reach {len(g)} different states")
    for b, s in states.items():
        if b == "G":
            continue
        if len(s) != 1:
            errs.append(f"{rec['task_id']}: behavior {b} not unique state")
        if s & g:
            errs.append(f"{rec['task_id']}: control behavior {b} reaches the same state as G")
    others = [next(iter(s)) for b, s in states.items() if b != "G" and s]
    if len(others) != len(set(others)):
        errs.append(f"{rec['task_id']}: two control behaviors share a state")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tss", required=True)
    ap.add_argument("--data", default="results/interface_invariance/benchmark.jsonl")
    a = ap.parse_args()
    import construct_paired_transitions as C
    by_dom, queries = L.load_tss(a.tss)
    n, errs = 0, []
    for line in open(a.data, encoding="utf-8"):
        rec = json.loads(line)
        execs = C.build_executors(rec, by_dom)
        errs += verify_record(rec, execs)
        n += 1
    print(f"verified {n} task records; {len(errs)} violations")
    for e in errs[:20]:
        print("  ", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
