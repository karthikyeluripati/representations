"""Environment validity checks (CPU). Writes results/env_validation.json.

python scripts/validate_environment.py --n 200
"""
import argparse, json, os, random, sys
from collections import Counter
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from environments.workflow_env import Env, make_task
from verifiers.verifiers import VERIFIERS, failure_mechanisms, noisy_control, strong
from agents import scripted as S

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()
out = {"n_tasks_per_family": args.n, "families": {}}
for fam in ("invoice", "inventory"):
    res = {}
    tasks = [make_task(fam, s) for s in range(args.n)]

    def run(policy, **kw):
        rows = []
        for i, t in enumerate(tasks):
            env = Env(t)
            rng = random.Random(f"{fam}:{policy.__name__}:{i}:{sorted(kw.items())}")
            if policy is S.correct:
                policy(env, **kw)
            else:
                policy(env, rng, **kw)
            fs = env.final_state()
            rows.append({v: VERIFIERS[v](t, fs) for v in VERIFIERS} | {"mech": failure_mechanisms(t, fs), "steps": fs["n_steps"]})
        return rows

    def summarize(rows):
        n = len(rows)
        d = {v: round(sum(r[v] for r in rows) / n, 3) for v in VERIFIERS}
        for w in ("w1_consistency", "w2_spotcheck"):
            acc = [r for r in rows if r[w]]
            d[f"{w}_false_pos_share_of_accepted"] = round(sum(not r["strong"] for r in acc) / len(acc), 3) if acc else None
            d[f"{w}_fp_mechanisms"] = dict(Counter(m for r in acc if not r["strong"] for m in r["mech"]))
        d["mean_steps"] = round(sum(r["steps"] for r in rows) / n, 1)
        return d

    for name, kw in S.CORRECT_VARIANTS.items():
        res[f"correct/{name}"] = summarize(run(S.correct, **kw))
    for name, fn in S.DIAGNOSTICS.items():
        res[f"diagnostic/{name}"] = summarize(run(fn))
    for p in (0.05, 0.1, 0.2):
        res[f"noisy_worker/p={p}"] = summarize(run(S.noisy_worker, p_skip=p, p_slip=p, p_stop=p / 2))
    out["families"][fam] = res

# single-step noise control: response-independent flips at a matched FP rate, on the p=0.1 noisy worker
fam = "invoice"
tasks = [make_task(fam, s) for s in range(args.n)]
nc = noisy_control(p_fp=0.3, p_fn=0.0, seed=1)
rows = []
for i, t in enumerate(tasks):
    env = Env(t); S.noisy_worker(env, random.Random(i), p_skip=0.1, p_slip=0.1, p_stop=0.05)
    fs = env.final_state(); rows.append((strong(t, fs), nc(t, fs)))
out["noise_control_invoice_p0.1"] = {"strong": round(sum(a for a, b in rows) / len(rows), 3), "noisy": round(sum(b for a, b in rows) / len(rows), 3),
                                     "fp_share_of_accepted": round(sum((not a) and b for a, b in rows) / max(1, sum(b for a, b in rows)), 3)}
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "env_validation.json"), "w"), indent=1)
for fam, res in out["families"].items():
    for k, v in res.items():
        print(fam[:4], k.ljust(32), "S", v["strong"], "W1", v["w1_consistency"], "W2", v["w2_spotcheck"], "FP|W1", v["w1_consistency_false_pos_share_of_accepted"], "FP|W2", v["w2_spotcheck_false_pos_share_of_accepted"])
print("noise control", out["noise_control_invoice_p0.1"])
