"""Arm runner: data assembly -> model/LoRA -> Trainer -> saved adapter + run record.

Every run writes results/runs/<arm>_s<seed>/{config.json, data_summary.json, train_log.jsonl, adapter/ (or merged/)}.
data_summary.json records example counts, dropped-by-length counts, target tokens and total tokens, and the sha256 of
the source data files, so matched-budget claims can be checked.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import random

import torch

from src.baselines import arms as AR
from src.environment.bfcl_env import load_bfcl, openai_tools
from src.training.render import HFRenderer, ToyRenderer
from src.training.trainer import Encoder, Trainer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAIRS = os.path.join(ROOT, "data", "cacrt_pairs_v1")


def _jl(p):
    return [json.loads(l) for l in open(p)]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_context(tasks_path=None):
    tasks = json.load(open(tasks_path or os.path.join(ROOT, "data", "tasks_v1.json")))
    by_id = {t["id"]: t for t in tasks}
    _, _, docs = load_bfcl()
    cache = {}

    def tools_fn(tid):
        if tid not in cache:
            cache[tid] = openai_tools(by_id[tid], docs)
        return cache[tid]
    return tasks, by_id, tools_fn


POLICY = os.path.join(ROOT, "data", "cacrt_pairs_v1_policy")


def _twins(policy_correct):
    if policy_correct:
        return [a for a in _jl(os.path.join(POLICY, "train_anchors_twin_policy.jsonl")) if a["policy_correct"]]
    return _jl(os.path.join(PAIRS, "train_anchors_twin.jsonl"))


def build_records(arm, tasks, by_id, seed=0, pairs_source="surrogate", policy_correct_anchors=False):
    spec = AR.ARMS[arm]
    train = [t for t in tasks if t["usable"] and t["split"] == "train"]
    states = _jl(os.path.join(PAIRS, "train_states.jsonl"))
    rng = random.Random(seed)
    recs, groups = [], []
    for d in spec.get("data", []):
        if d == "s0":
            g = AR.s0_trajs(train)
        elif d == "outage":
            g = AR.outage_trajs(train)
        elif d == "repair_traj":
            g = AR.repair_trajs(by_id, states)
        elif d == "paft":
            g = AR.paft_style(train, by_id, states, alpha=0.3, seed=seed)
        elif d == "action_pos":
            g = AR.action_records(states)
        elif d == "action_twin":
            g = [{**a, "type": "action"} for a in AR.anchor_records(_twins(policy_correct_anchors))]
        elif d == "pairs" and pairs_source == "policy":
            g = _jl(os.path.join(POLICY, "train_policy_pairs.jsonl"))
        elif d == "pairs":
            g = AR.pair_records(states, verified=True)
        elif d == "pairs_unverified":
            g = AR.pair_records(states, verified=False)
        else:
            raise ValueError(d)
        groups.append(g)
    if spec.get("balance"):
        recs = AR.balance(groups, rng)
    else:
        recs = [x for g in groups for x in g]
    anchors = []
    if spec.get("anchor"):
        twins = _twins(policy_correct_anchors)
        if spec["anchor_data"] == "twin":
            rows = twins
        else:                                   # generic pool capped at the twin pool size (matched anchor diversity)
            rows = _jl(os.path.join(PAIRS, "train_anchors_generic.jsonl"))
            rows = random.Random(seed).sample(rows, min(len(rows), len(twins)))
        anchors = AR.anchor_records(rows)
    probe = AR.anchor_records(_jl(os.path.join(PAIRS, "train_anchors_twin.jsonl")))
    return recs, anchors, probe


def build_val():
    dev_states = _jl(os.path.join(PAIRS, "eval_states_dev.jsonl"))
    dev_twins = _jl(os.path.join(PAIRS, "dev_anchors_twin.jsonl"))
    corr = [{"task_id": s["task_id"], "msgs": s["messages"], "correct": s["pos_action"],
             "alternatives": list(s["neg_actions"].values())} for s in dev_states if s["retained_pairs"]]
    tw = {k: [{"task_id": a["task_id"], "msgs": a["messages"], "correct": a["action"], "alternatives": list(a["negs"].values())}
              for a in dev_twins if a["kind"] == k] for k in ("T-out", "T-clean")}
    return {"dev_corrective": corr, "dev_twin_out": tw["T-out"], "dev_twin_clean": tw["T-clean"]}


def load_model(args):
    """args.model: 'tiny' (random Qwen3, CPU tests) or a local HF path (base or merged pi0)."""
    if args.model == "tiny":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(args.seed)
        cfg = Qwen3Config(vocab_size=258, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=16, max_position_embeddings=65536, tie_word_embeddings=True)
        model = Qwen3ForCausalLM(cfg)
        return model, ToyRenderer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, HFRenderer(tok)


def attach_lora(model, args):
    from peft import LoraConfig, get_peft_model
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lc = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, lc)


def run_arm(arm, args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    spec = AR.ARMS[arm]
    out = os.path.join(args.out_root, f"{arm}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    tasks, by_id, tools_fn = load_context()
    recs, anchors, probe = build_records(arm, tasks, by_id, args.seed, args.pairs, args.policy_correct_anchors)
    if args.limit:
        recs, anchors, probe = recs[:args.limit], anchors[:args.limit], probe[:args.limit]
    model, renderer = load_model(args)
    if args.lora:
        model = attach_lora(model, args)
        ref_ctx = model.disable_adapter
    else:
        frozen = copy.deepcopy(model).eval()
        for p in frozen.parameters():
            p.requires_grad_(False)

        @contextlib.contextmanager
        def ref_ctx():                         # swap parameters with the frozen reference for the duration
            live = [p.data for p in model.parameters()]
            for p, q in zip(model.parameters(), frozen.parameters()):
                p.data = q.data
            try:
                yield
            finally:
                for p, d in zip(model.parameters(), live):
                    p.data = d
    enc = Encoder(renderer, tools_fn, args.max_len)
    P = enc.encode_all(recs); dropP = enc.dropped
    A = enc.encode_all(anchors); dropA = enc.dropped - dropP
    Q = enc.encode_all(probe) if not spec.get("anchor") else []
    val = build_val() if args.val else {}
    if val and (args.limit or args.val_limit):
        n = args.limit or args.val_limit
        val = {k: random.Random(0).sample(v, min(n, len(v))) for k, v in val.items()}

    def n_tok(E):
        if E and "c" in E[0]:
            return {"target": sum(e["c"]["n_target"] + e["r"]["n_target"] for e in E),
                    "total": sum(len(e["c"]["input_ids"]) + len(e["r"]["input_ids"]) for e in E)}
        return {"target": sum(e["n_target"] for e in E), "total": sum(len(e["input_ids"]) for e in E)}
    steps = args.steps or max(1, round(args.epochs * len(P) / args.batch_size))
    cfg = {"arm": arm, **spec, "seed": args.seed, "lr": args.lr, "beta": args.beta, "lambda": (args.lam if args.lam is not None else spec.get("lambda", 0.0)) if spec.get("anchor") else 0.0,
           "batch_size": args.batch_size, "anchor_batch_size": args.anchor_batch_size, "micro_batch": args.micro_batch,
           "steps": steps, "eval_every": args.eval_every, "probe_every": args.probe_every, "max_grad_norm": 1.0,
           "chosen_nll": args.chosen_nll, "pairs_source": args.pairs, "policy_correct_anchors": args.policy_correct_anchors, "model": args.model, "lora": args.lora, "lora_r": args.lora_r, "max_len": args.max_len}
    if "grad_mode" in spec:
        cfg["grad_mode"] = spec["grad_mode"]
    summ = {"n_primary": len(P), "n_anchor": len(A), "n_probe": len(Q), "dropped_primary_by_length": dropP,
            "dropped_anchor_by_length": dropA, "primary_tokens_per_epoch": n_tok(P), "anchor_tokens_per_pass": n_tok(A),
            "data_files_sha256": {f: _sha(os.path.join(PAIRS, f)) for f in sorted(os.listdir(PAIRS)) if f.endswith(".jsonl")},
            "n_by_kind": {}}
    for e in P:
        k = e["rec"].get("scenario") or e["rec"].get("neg") or e["rec"]["type"]
        summ["n_by_kind"][k] = summ["n_by_kind"].get(k, 0) + 1
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump(summ, open(os.path.join(out, "data_summary.json"), "w"), indent=1)
    if spec["objective"] is None:
        return cfg, summ, []
    tr = Trainer(model, ref_ctx, renderer.pad_id, cfg, P, A, Q, val, os.path.join(out, "train_log.jsonl"), args.device)
    tr.precompute_ref()
    hist = tr.train(enc if val else None)
    if args.save:
        if args.lora and args.merge_out:
            model.merge_and_unload().save_pretrained(args.merge_out)
            renderer.tok.save_pretrained(args.merge_out)
        elif args.lora:
            model.save_pretrained(os.path.join(out, "adapter"))
    return cfg, summ, hist


def add_args(ap):
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--chosen-nll", dest="chosen_nll", type=float, default=0.0)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    ap.add_argument("--anchor-batch-size", dest="anchor_batch_size", type=int, default=8)
    ap.add_argument("--micro-batch", dest="micro_batch", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=25)
    ap.add_argument("--probe-every", dest="probe_every", type=int, default=10)
    ap.add_argument("--max-len", dest="max_len", type=int, default=12288)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", dest="lora_r", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: cap records")
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--pairs", choices=["surrogate", "policy"], default="surrogate",
                    help="pair source: CPU surrogate-verified, or pi0-sampled and pi0-continuation-verified (GPU stage)")
    ap.add_argument("--policy-correct-anchors", dest="policy_correct_anchors", action="store_true",
                    help="keep only twin anchors that pi0 already answers correctly (greedy)")
    ap.add_argument("--val-limit", dest="val_limit", type=int, default=48, help="dev monitoring states per set (fixed sample)")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--merge-out", dest="merge_out", default=None)
    ap.add_argument("--out-root", dest="out_root", default=os.path.join(ROOT, "results", "runs"))
    return ap
