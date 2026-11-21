"""Does a cheap deterministic trace check remove the multi-step (composition) false positives?

process_check: every listed record id was read (via get_record/get_records) AND has a ledger entry. It uses only the
tool trace and the agent's own ledger (no source values, no hidden answers).
Writes results/process_check_ablation.json.  python scripts/process_check_ablation.py --n 400
"""
import argparse, json, os, random, sys
from collections import Counter
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from environments.workflow_env import Env, make_task
from verifiers.verifiers import w1_consistency, w2_spotcheck, strong, failure_mechanisms
from agents import scripted as S


def process_check(task, fs):
    ids = set(task.records)
    return set(fs["ledger"]) == ids and set(fs["read"]) >= ids


ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); args = ap.parse_args()
out = {}
for fam in ("invoice", "inventory"):
    for p in (0.05, 0.1, 0.2):
        rows = []
        for i in range(args.n):
            t = make_task(fam, i); env = Env(t)
            S.noisy_worker(env, random.Random(f"{fam}{p}{i}"), p_skip=p, p_slip=p, p_stop=p / 2)
            fs = env.final_state()
            rows.append({"strong": strong(t, fs), "w1": w1_consistency(t, fs), "w2": w2_spotcheck(t, fs),
                         "proc": process_check(t, fs), "mech": failure_mechanisms(t, fs)})
        res = {"strong_rate": round(sum(r["strong"] for r in rows) / len(rows), 3)}
        for name, acc in (("w1", lambda r: r["w1"]), ("w1+proc", lambda r: r["w1"] and r["proc"]),
                          ("w2", lambda r: r["w2"]), ("w2+proc", lambda r: r["w2"] and r["proc"])):
            a = [r for r in rows if acc(r)]
            fps = [r for r in a if not r["strong"]]
            res[name] = {"accept_rate": round(len(a) / len(rows), 3), "fp_share_of_accepted": round(len(fps) / max(1, len(a)), 3),
                         "fp_mechanisms": dict(Counter(m for r in fps for m in r["mech"]))}
        out[f"{fam}/p={p}"] = res
        print(fam[:4], p, "strong", res["strong_rate"], "FP|W1", res["w1"]["fp_share_of_accepted"], "FP|W1+proc", res["w1+proc"]["fp_share_of_accepted"],
              "residual mech", res["w1+proc"]["fp_mechanisms"])
json.dump(out, open(os.path.join(ROOT, "results", "process_check_ablation.json"), "w"), indent=1)
