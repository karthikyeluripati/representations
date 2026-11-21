"""Fail loudly unless a data manifest matches expectations (used by the pod run scripts).
  python scripts/check_manifest.py MANIFEST.json key=value ...                 (top-level integer fields; 'dropped' must be empty)
  python scripts/check_manifest.py MANIFEST_render.json --file NAME=path --n NAME=count   (rendered files: sha256 and size)"""
import argparse
import hashlib
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("expect", nargs="*")
ap.add_argument("--file", action="append", default=[]); ap.add_argument("--n", action="append", default=[])
a = ap.parse_args()
m = json.load(open(a.manifest))
for kv in a.expect:
    k, v = kv.split("=")
    if m.get(k) != int(v):
        sys.exit(f"ABORT: {a.manifest}: {k}={m.get(k)} != {v}")
if "dropped" in m and m["dropped"]:
    sys.exit(f"ABORT: {a.manifest}: dropped={m['dropped']}")
for kv in a.file:
    k, p = kv.split("=", 1)
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != m[k]["sha256"]:
        sys.exit(f"ABORT: {p} sha256 differs from {a.manifest}")
for kv in a.n:
    k, v = kv.split("=")
    if m[k]["n"] != int(v):
        sys.exit(f"ABORT: {k} n={m[k]['n']} != {v}")
print("manifest ok:", a.manifest)
