"""Re-run the parent project's BFCL validation (task build, solvability, budgets, leakage, determinism, error pools)
and its data construction, then assert byte-identity with the locally validated manifest. Wrapper; no new logic."""
import json
import os
import subprocess
import sys

import _common

P = _common.PROJ
env = dict(os.environ, PYTHONHASHSEED="0")
subprocess.run([sys.executable, os.path.join(P, "scripts", "validate_environments.py")], check=True, env=env, stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, os.path.join(P, "scripts", "construct_correction_pairs.py")], check=True, env=env, stdout=subprocess.DEVNULL)
m = json.load(open(os.path.join(P, "data", "cacrt_pairs_v1", "MANIFEST.json")))
r = json.load(open(os.path.join(P, "data", "cacrt_pairs_v1", "MANIFEST.local.json")))
v = json.load(open(os.path.join(P, "results", "validation", "environment_validation.json")))
ok = m["files"] == r["files"] and m["tasks_file_sha256"] == r["tasks_file_sha256"] == v["tasks_sha256"]
print(json.dumps({"hashes_match": ok, "tasks_sha256": v["tasks_sha256"], "unsolvable": v["V2_unsolvable_task_ids"],
                  "leakage": v["V4_leakage"], "determinism_failures": v["V5_determinism_failures_first60"]}, indent=1))
sys.exit(0 if ok else 1)
