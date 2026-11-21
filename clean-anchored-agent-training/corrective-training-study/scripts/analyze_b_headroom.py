"""Headroom analysis for verified DPO (B): which held-out failures are stable across seeds, which are competence-limited
(the same task fails fault-free), and which are recovery-specific. Usage: python analyze_b_headroom.py <replication/eval dir> [heldout|test]"""
import collections, json, os, sys

EV = sys.argv[1]
STUDY = r"C:\Users\karth\OneDrive\Desktop\projects\representations\clean-anchored-agent-training\corrective-training-study"
PROJ = os.path.dirname(STUDY)
sys.path.insert(0, os.path.join(STUDY, "scripts")); sys.path.insert(0, PROJ)
import types, importlib.abc, importlib.machinery
from unittest.mock import MagicMock
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):   # bfcl_eval only needed for the env, not for labelling
    def find_spec(self, name, path, target=None):
        return importlib.machinery.ModuleSpec(name, self, is_package=True) if name.split('.')[0] == 'bfcl_eval' else None
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__file__ = '/nonexistent/bfcl_eval/__init__.py'; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, _Stub())
import analyze_failure_types as F  # noqa

tasks = {t["id"]: t for t in json.load(open(os.path.join(PROJ, "data", "tasks_v1.json"), encoding="utf-8"))}
def load(n): return [json.loads(l) for l in open(os.path.join(EV, n + ".jsonl"), encoding="utf-8")]
key = lambda r: (r["task_id"], r["suite"], r["scenario"], r["seed"])
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "heldout"
def keyed(rows):
    seen = collections.Counter(); out = {}
    for r in rows:
        if r["split"] != SPLIT: continue
        k = key(r); out[k + (seen[k],)] = r; seen[k] += 1     # RP/RA reuse (task, suite, scenario, seed): add occurrence index
    return out
runs = {n: keyed(load(n)) for n in ["pi0", "A_s0", "A_s1", "A_s2", "B_s0", "B_s1", "B_s2"]}
K = sorted(runs["pi0"]); assert all(set(v) == set(K) for v in runs.values()), "episode sets differ"
print(SPLIT, "episodes", len(K))
def prim(r):
    if r["correct"]: return "ok"
    L = F.labels(r, tasks[r["task_id"]]); return next((x for x in F.ORDER if x in L), "G")
lab = {n: {k: prim(r) for k, r in v.items()} for n, v in runs.items()}
Bn, An = ["B_s0", "B_s1", "B_s2"], ["A_s0", "A_s1", "A_s2"]
nB = {k: sum(runs[n][k]["correct"] for n in Bn) for k in K}
nA = {k: sum(runs[n][k]["correct"] for n in An) for k in K}
p0 = {k: runs["pi0"][k]["correct"] for k in K}

# 1. per-suite success
suites = sorted({k[1] for k in K})
print("\n1. success by suite (pi0 | A mean | B mean | n)")
for s in suites:
    ks = [k for k in K if k[1] == s]
    print(f"  {s:7s} {sum(p0[k] for k in ks)/len(ks):.3f} | {sum(nA[k] for k in ks)/3/len(ks):.3f} | {sum(nB[k] for k in ks)/3/len(ks):.3f} | {len(ks)}")

# 2. seed agreement of B
print("\n2. B: episodes by #seeds correct", collections.Counter(nB.values()), " A:", collections.Counter(nA.values()))

# 3. B's stable failures (0/3 correct) by label (seed-0 label) x whether pi0 also failed
core = [k for k in K if nB[k] == 0]
print(f"\n3. B fails in all 3 seeds: {len(core)}; pi0 also fails: {sum(not p0[k] for k in core)}; A fails all 3 too: {sum(nA[k]==0 for k in core)}")
c = collections.Counter((lab['B_s0'][k], "pi0_fail" if not p0[k] else "pi0_ok") for k in core)
for (l, p), v in sorted(c.items(), key=lambda x: -x[1]): print(f"   {l} {p}: {v}")

# 4. competence ceiling: is the same task's fault-free (S0) episode also failed by B?
s0B = {k[0]: nB[k] for k in K if k[1] == "S0" and k[4] == 0}
nonS0 = [k for k in core if k[1] != "S0"]
cap = collections.Counter((lab['B_s0'][k], "S0_fails" if s0B.get(k[0], 3) == 0 else "S0_ok") for k in nonS0)
print(f"\n4. B core failures outside S0: {len(nonS0)}; whose task B also cannot solve fault-free (S0 0/3): {sum(s0B.get(k[0],3)==0 for k in nonS0)}")
for (l, p), v in sorted(cap.items(), key=lambda x: -x[1]): print(f"   {l} {p}: {v}")
fixable = [k for k in nonS0 if s0B.get(k[0], 3) == 3]
print(f"   -> failures where B solves the same task fault-free in 3/3 seeds (recovery-specific): {len(fixable)}",
      collections.Counter((k[1], lab['B_s0'][k]) for k in fixable).most_common(12))

# 5. regressions and gains vs pi0 (majority over seeds)
reg = [k for k in K if p0[k] and nB[k] <= 1]; gain = [k for k in K if not p0[k] and nB[k] >= 2]
print(f"\n5. B vs pi0: gains {len(gain)}, regressions {len(reg)}")
print("   regressions by (suite,label):", collections.Counter((k[1], lab['B_s0'][k]) for k in reg).most_common(10))
print("   gains by (suite, pi0 label):", collections.Counter((k[1], lab['pi0'][k]) for k in gain).most_common(10))

# 6. S3 (impossible) behaviour: what does B do instead of escalating?
s3 = [k for k in K if k[1] == "S3"]
def s3kind(r):
    if r["escalated"]: return "escalated"
    if r["hit_step_cap"]: return "step_cap"
    last = r["completions"][-1] if r["completions"] else ""
    return "stopped_text:" + ("Done" if last.strip().startswith("Done") else "other")
for n in ["pi0", "A_s0", "B_s0"]:
    print(f"\n6. S3 {n}:", collections.Counter(s3kind(runs[n][k]) for k in s3),
          " mean calls", round(sum(runs[n][k]["n_calls"] for k in s3) / len(s3), 2))
ex = next(runs["B_s0"][k] for k in s3 if not runs["B_s0"][k]["escalated"])
print("   example B S3 miss:", ex["task_id"], [(c[0], c[2][:70]) for c in ex["calls"]][-4:], "| last text:", ex["completions"][-1][:120])

# 7. B's argument (C) failures: same wrong call as pi0?
cB = [k for k in core if lab["B_s0"][k] == "C"]
same = sum(json.dumps(runs["B_s0"][k]["calls"])[:4000] == json.dumps(runs["pi0"][k]["calls"])[:4000] for k in cB)
print(f"\n7. B core argument failures {len(cB)}; identical call sequence to pi0: {same}; by suite", collections.Counter(k[1] for k in cB))
