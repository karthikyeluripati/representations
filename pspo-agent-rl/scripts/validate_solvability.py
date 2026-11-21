"""Build paired S0-S3 tasks from BFCL v4 multi_turn_base, verify solvability with the executable reference policy, and
write the split manifest. python scripts/validate_solvability.py --out data/tasks.json"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from environments.bfcl_faults import load_bfcl, build_task, Episode, reference_policy, new_instances, public_state, run_calls

HELDOUT_CLASSES = {"VehicleControlAPI"}   # unseen tool families (test only)

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/tasks.json"); args = ap.parse_args()
q, a, docs = load_bfcl()
tasks, skipped, report = [], collections.Counter(), collections.Counter()
for e in q:
    for k in ("initial_config", "involved_classes", "excluded_function", "question", "path"):
        if isinstance(e.get(k), str):
            e[k] = json.loads(e[k]) if e[k].strip().startswith(("{", "[")) and '"' in e[k] else eval(e[k])  # noqa: S307 (trusted BFCL file)
    for turn in range(len(a[e["id"]])):
        t = build_task(e, a[e["id"]], turn)
        if t is None:
            skipped["no_state_change_or_no_user_text"] += 1; continue
        ok = True
        for sc in ("S0", "S1", "S2", "S3"):
            for seed in (0, 1):
                ep = Episode(t, sc, seed); reference_policy(ep); g = ep.grade()
                report[(sc, g["correct"])] += 1
                ok &= g["correct"]
        # impossibility check for S3: the fault class must need a state change, and without its methods it cannot change
        start = new_instances(e); run_calls(start, t["prefix_gt"]); init_c = public_state(start[t["fault_class"]])
        final = new_instances(e); run_calls(final, t["prefix_gt"] + t["turn_gt"])
        needs_change = public_state(final[t["fault_class"]]) != init_c
        if not (ok and needs_change):
            skipped["reference_failed_or_no_change"] += 1; continue
        t["split"] = "test_heldout_family" if t["fault_class"] in HELDOUT_CLASSES else None
        tasks.append(t)
# split seen-family tasks 60/20/20 by task id hash (pairs stay together: all scenarios of a task share its split)
import hashlib
for t in tasks:
    if t["split"] is None:
        r = int(hashlib.md5(t["base_id"].encode()).hexdigest(), 16) % 10   # all turns of a BFCL task share a split
        t["split"] = "train" if r < 6 else ("dev" if r < 8 else "test")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump([{k: v for k, v in t.items()} for t in tasks], open(args.out, "w"), default=str)
print("tasks", len(tasks), "skipped", dict(skipped))
print("reference outcomes", {f"{k[0]}:{k[1]}": v for k, v in sorted(report.items())})
print("by split", dict(collections.Counter(t["split"] for t in tasks)), "by fault class", dict(collections.Counter(t["fault_class"] for t in tasks)))
