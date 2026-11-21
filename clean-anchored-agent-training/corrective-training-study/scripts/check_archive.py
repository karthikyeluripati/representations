"""Verify the H100 gate archive and recover the inputs this study needs (read-only on the archive).

Checks:
  1. archive sha256 == recorded value;
  2. every file listed in the archived data manifests hashes correctly (read in memory, nothing extracted);
  3. the archived base / pi0 evaluation summaries are RECOMPUTED from the archived per-episode traces with the
     current metric code and must match to 1e-12 (no stale or edited summaries);
  4. the 567 policy-verified pairs: count, distinct states, negative types, train-only task ids, a+ / a- success
     counts consistent with the stored K=4 rollouts and the verify_stochastic rule.
Recovers data/policy_pairs_v1.jsonl (byte copy of the archived train_policy_pairs.jsonl; sha256 re-checked).
Writes results/archive_check.json.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))
sys.path.insert(0, PROJ)

from src.counterfactuals.construct import verify_stochastic  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402

ARCHIVE_SHA = "a460d0fa7134acc61e84db32e2df610b8fe149430e3cc33b358adaa3a8276c18"
PFX = "clean-anchored-agent-training/"
HEAD = ["fault_free", "genuine_repair", "injected_recovery", "conditional_recovery", "correct_escalation",
        "premature_abstention", "false_outage_rate", "overall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join(PROJ, "..", "pod_backups", "pod_backup_20261119.tgz"))
    a = ap.parse_args()
    rep = {"archive": os.path.abspath(a.archive)}
    raw = open(a.archive, "rb").read()
    rep["archive_sha256_ok"] = hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA
    t = tarfile.open(fileobj=io.BytesIO(raw))
    rd = lambda n: t.extractfile(PFX + n).read()  # noqa: E731
    for mf, d in (("data/cacrt_pairs_v1/MANIFEST.json", "data/cacrt_pairs_v1/"), ("data/cacrt_pairs_v1_policy/MANIFEST.json", "data/cacrt_pairs_v1_policy/")):
        m = json.loads(rd(mf))
        rep[f"manifest_ok:{mf}"] = all(hashlib.sha256(rd(d + f)).hexdigest() == h for f, h in m["files"].items())
    rep["tasks_sha256"] = hashlib.sha256(rd("data/tasks_v1.json")).hexdigest()
    local_tasks = os.path.join(PROJ, "data", "tasks_v1.json")
    rep["local_tasks_match_archive"] = hashlib.sha256(open(local_tasks, "rb").read()).hexdigest() == rep["tasks_sha256"]
    tasks = {x["id"]: x for x in json.loads(rd("data/tasks_v1.json"))}
    # 3. recompute summaries
    rep["summaries"] = {}
    for name in ("base", "pi0"):
        rows = [json.loads(l) for l in io.BytesIO(rd(f"results/eval/{name}.jsonl"))]
        saved = json.loads(rd(f"results/eval/{name}.summary.json"))
        ok = True
        for sp in ("test", "heldout"):
            s = M.summarize([r for r in rows if r["split"] == sp])
            for k in HEAD:
                if (s[k] is None) != (saved[sp][k] is None) or (s[k] is not None and abs(s[k] - saved[sp][k]) > 1e-12):
                    ok = False
            rep["summaries"][f"{name}:{sp}"] = {k: s[k] for k in HEAD} | {"n": s["n"], "n_by_suite": s["n_by_suite"],
                                                                          "correct_by_suite": {k: sum(r["correct"] for r in rows if r["split"] == sp and r["suite"] == k) for k in s["n_by_suite"]}}
        rep[f"summary_recomputed_matches:{name}"] = ok
    # 4. pairs
    blob = rd("data/cacrt_pairs_v1_policy/train_policy_pairs.jsonl")
    pairs = [json.loads(l) for l in io.BytesIO(blob)]
    neg = {}
    for p in pairs:
        k = p["neg"].split("+")[0] if p["neg"].startswith("cpu_") else "sampled"
        neg[k] = neg.get(k, 0) + 1
    rep["pairs"] = {"n": len(pairs), "distinct_states": len({p["state"] for p in pairs}), "by_negative": neg,
                    "all_train_tasks": all(tasks[p["task_id"]]["split"] == "train" for p in pairs),
                    "all_pass_verify_rule": all(verify_stochastic(p["succ_pos"], p["succ_neg"])[0] for p in pairs),
                    "rollouts_per_branch": sorted({len(p["succ_pos"]) for p in pairs} | {len(p["succ_neg"]) for p in pairs}),
                    "chosen_success_total": sum(sum(p["succ_pos"]) for p in pairs), "rejected_success_total": sum(sum(p["succ_neg"]) for p in pairs),
                    "official_cso_rule_would_keep": sum(1 for p in pairs if any(p["succ_pos"]))}
    out = os.path.join(STUDY, "data"); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "policy_pairs_v1.jsonl"), "wb").write(blob)
    rep["recovered"] = {"data/policy_pairs_v1.jsonl": hashlib.sha256(open(os.path.join(out, "policy_pairs_v1.jsonl"), "rb").read()).hexdigest()}
    os.makedirs(os.path.join(STUDY, "results"), exist_ok=True)
    json.dump(rep, open(os.path.join(STUDY, "results", "archive_check.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "summaries"}, indent=1))


if __name__ == "__main__":
    main()
