"""Dry-run of the pre-registered pilot gates on scripted stand-in runs (no model outputs exist)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); STUDY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(STUDY, "src"))
from verification import semantic_cf as T1  # noqa: E402
from verification import semantic_cf_t2 as T2  # noqa: E402


def _write(d, name, rows):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _run(tmp, stand_in):
    t2 = [s for s in T2.specs("test") if s["family"] in (40, 41, 42, 43)]
    dev = [s for s in T2.specs("dev") if s["family"] in (30, 31)]
    t1 = [s for s in T1.specs("test") if s["family"] in (20, 21, 22)]
    for name, pol in stand_in.items():
        _write(str(tmp / "t2"), name, T2.run_policy(pol[0](), t2))
        _write(str(tmp / "t1"), name, T1.run_policy(pol[1](), t1, T1.SYSTEM))
    for pid, pol in (("P0", T2.T2AlwaysRetry), ("P2", T2.T2AlwaysCheck)):
        _write(str(tmp / "dev"), f"piB@{pid}", T2.run_policy(pol(), dev)); _write(str(tmp / "dev"), f"CAW2_s0@{pid}", T2.run_policy(pol(), dev))
        _write(str(tmp / "t2"), f"piB@{pid}", T2.run_policy(pol(), t2)); _write(str(tmp / "t1"), f"piB@{pid}", T1.run_policy(T1.AlwaysCheck(), t1, T1.SYSTEM))
        for s in ("CAW2_s0", "CAW2_s1"):
            _write(str(tmp / "t2"), f"{s}@{pid}", T2.run_policy(pol(), t2)); _write(str(tmp / "t1"), f"{s}@{pid}", T1.run_policy(T1.AlwaysCheck(), t1, T1.SYSTEM))
    out = tmp / "a.json"
    subprocess.run([sys.executable, os.path.join(STUDY, "scripts", "cf_pilot_analysis.py"), "--stage", "final", "--t2", str(tmp / "t2"),
                    "--t2dev", str(tmp / "dev"), "--t1", str(tmp / "t1"), "--out", str(out)], check=True, capture_output=True)
    return json.load(open(out))


def test_gates_dry_run(tmp_path):
    good = (T2.T2ContractPolicy, T1.ContractPolicy); caw = (T2.T2AlwaysCheck, T1.AlwaysCheck)
    stand = {**{f"CF_s{i}": good for i in range(3)}, **{f"CTRL_s{i}": caw for i in range(3)}, **{f"ABL_s{i}": caw for i in range(2)},
             **{f"CAW2_s{i}": caw for i in range(2)}}
    r = _run(tmp_path, stand)
    g = r["gates"]
    assert g["S1"] and g["S2"] and g["S3"] and g["S5"] and not g["S6_prompt_matches_CF"]
    assert r["S4_detail"]["regression_available"] is False and r["decision"].startswith("NO CLAIM")   # S4 needs regression traces
    assert r["prompt_baselines"]["piB"]["prompt"] == "P2"            # dev selection picks the better dev prompt
