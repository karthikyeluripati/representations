"""Mechanism analyses.

--mode premise   (run BEFORE the pilot, on pi0; ~10 GPU-min) Tests the premise CACRT rests on: corrective DPO gradients
                 conflict more with the counterfactual-twin anchors than with generic clean anchors. Reports, over R
                 random batches, cos(g_DPO, g_twin_out), cos(g_DPO, g_twin_clean), cos(g_DPO, g_generic), and the
                 first-order predicted change of each anchor loss under one DPO step. If twin and generic conflicts are
                 indistinguishable, twin anchoring has no mechanistic reason to beat generic replay (pre-registered STOP
                 signal for the twin-anchor component).
--mode curves    (after training) Per-arm curves from results/runs/*/train_log.jsonl: dev decision accuracy on corrective
                 states and twin anchors (retention) vs step, and gradient cosines (cos_PA, or probe_cos for arms without
                 anchors). Writes results/analysis/curves.json. Cosines are descriptive; causal claims rest only on matched
                 arm comparisons (EXP3 vs B7_replay vs EXP2 vs EXP1).
"""
import argparse
import glob
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def premise(args):
    import torch
    from src.training import objectives as O
    from src.training.run import _jl, PAIRS, attach_lora, build_records, load_context, load_model
    from src.training.trainer import Encoder, Trainer
    from src.baselines.arms import anchor_records
    if args.model != "tiny" and not args.lora:
        raise SystemExit("premise on a real model needs --lora (policy = pi0 + zero-init adapter, ref = pi0)")
    torch.manual_seed(0)
    tasks, by_id, tools_fn = load_context()
    model, renderer = load_model(args)
    if args.lora:
        model = attach_lora(model, args)
        ref = model.disable_adapter
    else:                                              # CPU smoke: reference == policy at the evaluated point
        import contextlib
        ref = contextlib.nullcontext
    enc = Encoder(renderer, tools_fn, args.max_len)
    pairs, _, _ = build_records("EXP2", tasks, by_id)
    sets = {"twin_out": [a for a in anchor_records(_jl(os.path.join(PAIRS, "train_anchors_twin.jsonl"))) if a["kind"] == "T-out"],
            "twin_clean": [a for a in anchor_records(_jl(os.path.join(PAIRS, "train_anchors_twin.jsonl"))) if a["kind"] == "T-clean"],
            "generic": anchor_records(_jl(os.path.join(PAIRS, "train_anchors_generic.jsonl")))}
    rng = random.Random(0)
    P = enc.encode_all(rng.sample(pairs, min(args.n, len(pairs))))
    E = {k: enc.encode_all(rng.sample(v, min(args.n, len(v)))) for k, v in sets.items()}
    cfg = {"objective": "dpo", "lr": 1e-5, "beta": 0.1, "batch_size": args.batch, "steps": 1, "anchor": "nll"}
    tr = Trainer(model, ref, renderer.pad_id, cfg, P, device=args.device)
    tr.precompute_ref()
    model.train()                                      # activates gradient checkpointing (as in training)
    res = {k: [] for k in E}
    for r in range(args.rounds):
        pb = rng.sample(P, min(args.batch, len(P)))
        gP, _, _ = tr._grad(tr.primary_loss, pb, 1)
        for k, ex in E.items():
            gA, _, _ = tr._grad(tr.anchor_loss, rng.sample(ex, min(args.batch, len(ex))), 1)
            # first-order change of anchor loss under a descent step on DPO: dL_A ~ -lr * gA . gP
            res[k].append({"cos": O.cosine(gP, gA), "dot": float(gA @ gP)})
    out = {k: {"mean_cos": sum(x["cos"] for x in v) / len(v), "frac_conflict": sum(x["dot"] < 0 for x in v) / len(v),
               "rounds": v} for k, v in res.items()}
    od = os.path.join(ROOT, "results", "analysis"); os.makedirs(od, exist_ok=True)
    json.dump({"model": args.model, **out}, open(os.path.join(od, f"premise_{os.path.basename(args.model)}.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rounds"} for k, v in out.items()}, indent=1))


def curves(args):
    out = {}
    for f in sorted(glob.glob(os.path.join(args.runs, "*", "train_log.jsonl"))):
        name = os.path.basename(os.path.dirname(f))
        rows = [json.loads(l) for l in open(f)]
        out[name] = {"decision": [(r["step"], r["decision"]) for r in rows if "decision" in r],
                     "cos": [(r["step"], r.get("cos_PA", r.get("probe_cos"))) for r in rows if ("cos_PA" in r or "probe_cos" in r)],
                     "loss": [(r["step"], r.get("loss_primary"), r.get("loss_anchor")) for r in rows if "loss_primary" in r]}
    od = os.path.join(ROOT, "results", "analysis"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "curves.json"), "w"), indent=1)
    for k, v in out.items():
        last = v["decision"][-1][1] if v["decision"] else None
        first = v["decision"][0][1] if v["decision"] else None
        print(k, "decision first->last:", first, "->", last)


def main():
    from src.training.run import add_args
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--mode", choices=["premise", "curves"], required=True)
    ap.add_argument("--runs", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=16)
    a = ap.parse_args()
    premise(a) if a.mode == "premise" else curves(a)


if __name__ == "__main__":
    main()
