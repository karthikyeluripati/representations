"""Compile the benchmark leaderboard (paper Table 1) from the verified result archives.

Metric per wording: AC success given S0, the mean of ACa and ACl over the 116 diagnostic BFCL tasks the policy
solves fault-free (std prompt unless marked). Trained policies are averaged over seeds (per-seed values kept).
tau2 column: check rate right after the implicit error on the 13 diagnostic airline writes (pooled seeds).
Usage: python scripts/compile_leaderboard.py --uo DIR --synth DIR --tau2 FILE --syn64 FILE --out paper/tables
"""
import argparse
import csv
import json
import os

import numpy as np

W = ("explicit", "para", "implicit")


def row_from_summary(s, prompt="std"):
    p = s[prompt]
    return {"S0": p["S0"]["success"], "S1|S0": p["S1"]["success_given_S0"], **{w: p[f"AC_{w}_given_S0"] for w in W}}


def mean_rows(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, {k: [round(r[k], 3) for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uo", required=True); ap.add_argument("--synth", required=True); ap.add_argument("--tau2", required=True)
    ap.add_argument("--syn64", required=True, help="json with per-seed SYN64 summaries (recorded from the pod)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = lambda d, n: json.load(open(os.path.join(d, n + ".summary.json")))
    T = []
    untrained = [("Qwen3-4B-Instruct-2507 (base)", "base4b"), ("Qwen3-8B (no thinking)", "q8b"), ("Qwen3-30B-A3B-Instruct-2507 (FP8)", "q30b_a3b"),
                 ("π₀ (corrective SFT)", "pi0"), ("B: CSO-style verified DPO", "B_s0")]
    for name, f in untrained:
        s = L(a.uo, f)
        for pr in ("std", "check"):
            if pr in s:
                T.append({"policy": name + ("" if pr == "std" else " + check prompt"), "group": "untrained / standard training",
                          "seeds": 1, **row_from_summary(s, pr)})
    tau = json.load(open(a.tau2))["diagnostic|implicit"]["arms"]
    syn64 = json.load(open(a.syn64))
    for name, arm, grp in (("SYN-old (synthetic APIs, retry convention)", "SYNOLD", "this work: controls"),
                           ("SYN-32 (32 synthetic classes, IAR)", "SYNIAR", "this work"),
                           ("SYN-64 (64 synthetic classes, IAR)", "SYN64IAR", "this work")):
        rows = [row_from_summary(json.load(open(os.path.join(a.synth, f"{arm}_s{s}.summary.json")))) for s in (0, 1, 2)] if arm != "SYN64IAR" \
            else [{"S0": x["S0"], "S1|S0": x["S1|S0"], **{w: x[w] for w in W}} for x in syn64]
        m, per = mean_rows(rows)
        T.append({"policy": name, "group": grp, "seeds": 3, **m, "per_seed": per, "tau2_check": tau.get(arm, {}).get("check")})
    T.append({"policy": "scripted verify-then-decide (reference)", "group": "reference", "seeds": "-", "S0": 1.0, "S1|S0": 1.0,
              "explicit": 1.0, "para": 1.0, "implicit": 1.0, "tau2_check": 1.0})
    T.append({"policy": "scripted blind retry (reference)", "group": "reference", "seeds": "-", "S0": 1.0, "S1|S0": 1.0,
              "explicit": 0.5, "para": 0.5, "implicit": 0.5, "tau2_check": 0.0})
    os.makedirs(a.out, exist_ok=True)
    cols = ["policy", "group", "seeds", "S0", "S1|S0", "explicit", "para", "implicit", "tau2_check"]
    with open(os.path.join(a.out, "leaderboard.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(T)
    fmt = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    md = ["| policy | seeds | S0 | S1\\|S0 | explicit | para | implicit | τ² check (implicit) |", "|---|---|---|---|---|---|---|---|"]
    for r in T:
        md.append("| " + " | ".join([r["policy"], str(r["seeds"])] + [fmt(r.get(c)) for c in cols[3:]]) + " |")
    open(os.path.join(a.out, "leaderboard.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    json.dump(T, open(os.path.join(a.out, "leaderboard.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
