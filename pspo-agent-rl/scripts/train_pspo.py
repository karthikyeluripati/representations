"""LoRA training with matched data for all objectives (same init, same trajectories, same pairs, 1 epoch).

  --objective sft            : B7  SFT on reference trajectories of all scenarios S0-S3 (paired examples)
  --objective sft_recovery   : B4  SFT on S0-S2 trajectories only (successful recovery examples)
  --objective sft_samectx    : B7p SFT + DPO-style SAME-context preference at divergence points
                               (at h_imp prefer a_imp over a_solv; at h_solv prefer a_solv over a_imp)
  --objective pspo           : SFT + CROSS-context paired-solvability contrast at divergence points
                               (the same action must become more likely in the world whose evidence supports it:
                                lp(a_imp|h_imp) - lp(a_imp|h_solv) and lp(a_solv|h_solv) - lp(a_solv|h_imp), relative to the reference model)
B7p and PSPO use exactly the same four log-probabilities per pair; only the contrast direction differs.
"""
import argparse, json, math, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from environments.bfcl_faults import load_bfcl
from training.paired_data import build, assistant_indices

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--objective", required=True, choices=["sft", "sft_recovery", "sft_samectx", "pspo"])
ap.add_argument("--out", required=True); ap.add_argument("--tasks", default="data/tasks.json")
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)

tok = AutoTokenizer.from_pretrained(args.model)
tasks = [t for t in json.load(open(args.tasks)) if t["split"] == "train"]
_, _, docs = load_bfcl()
trajs, pairs = build(tasks, docs, seeds=(0,))
if args.objective == "sft_recovery":
    trajs = [t for t in trajs if t["scenario"] != "S3"]
use_pairs = args.objective in ("sft_samectx", "pspo")


def render(msgs, tools, gen=False):
    return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=False)


def ids(s):
    return tok(s, add_special_tokens=False)["input_ids"]


def sft_example(tr):
    full = ids(render(tr["msgs"], tr["tools"]))
    labels = [-100] * len(full)
    for i in assistant_indices(tr):
        p = ids(render(tr["msgs"][:i], tr["tools"], gen=True))
        q = ids(render(tr["msgs"][: i + 1], tr["tools"]))
        assert full[: len(q)] == q and q[: len(p)] == p, "chat template not prefix-stable"
        for j in range(len(p), len(q)):
            labels[j] = full[j]
    return full, labels


def ctx_action(tr, i):
    p = ids(render(tr["msgs"][:i], tr["tools"], gen=True))
    q = ids(render(tr["msgs"][: i + 1], tr["tools"]))
    return p, q[len(p):]


model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)


def seq_lp(ctx, act, ref=False):
    x = torch.tensor([ctx + act], device="cuda")
    if ref:
        with torch.no_grad(), model.disable_adapter():
            logits = model(x).logits
    else:
        logits = model(x).logits
    lp = F.log_softmax(logits[0, len(ctx) - 1: -1].float(), -1)
    return lp.gather(-1, torch.tensor(act, device="cuda")[:, None]).sum()


items = [("sft", t) for t in trajs] + ([("pair", p) for p in pairs] if use_pairs else [])
random.shuffle(items)
if args.max_steps:
    items = items[: args.max_steps]
os.makedirs(args.out, exist_ok=True)
log = open(os.path.join(args.out, "train_log.jsonl"), "w")
t0, n_tok = time.time(), 0
model.train()
total = len(items); warm = max(1, total // 20)
for step, (kind, it) in enumerate(items):
    lr = args.lr * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * step / total)))
    for g in opt.param_groups:
        g["lr"] = lr
    if kind == "sft":
        full, labels = sft_example(it)
        x = torch.tensor([full], device="cuda"); y = torch.tensor([labels], device="cuda")
        out = model(input_ids=x, labels=y)
        loss = out.loss; rec = {"step": step, "kind": "sft", "loss": loss.item(), "scenario": it["scenario"]}
        n_tok += len(full)
    else:
        hs, as_ = ctx_action(it["solvable"], it["i_solvable"])
        hi, ai = ctx_action(it["impossible"], it["i_impossible"])
        lp = {k: seq_lp(c, a) for k, (c, a) in {"ss": (hs, as_), "si": (hs, ai), "ii": (hi, ai), "is": (hi, as_)}.items()}
        rf = {k: seq_lp(c, a, ref=True) for k, (c, a) in {"ss": (hs, as_), "si": (hs, ai), "ii": (hi, ai), "is": (hi, as_)}.items()}
        d = {k: lp[k] - rf[k] for k in lp}
        if args.objective == "sft_samectx":   # same context, different actions
            m1 = d["ii"] - d["is"]; m2 = d["ss"] - d["si"]
        else:                                  # pspo: same action, different contexts (evidence sensitivity)
            m1 = d["ii"] - d["si"]; m2 = d["ss"] - d["is"]
        loss = args.lam * (-F.logsigmoid(args.beta * m1) - F.logsigmoid(args.beta * m2))
        rec = {"step": step, "kind": "pair", "loss": loss.item(), "m1": m1.item(), "m2": m2.item(), "type": it["type"]}
        n_tok += 2 * (len(hs) + len(hi))
    (loss / args.accum).backward()
    if (step + 1) % args.accum == 0 or step == total - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
    rec.update({"lr": lr, "elapsed": round(time.time() - t0, 1)})
    log.write(json.dumps(rec) + "\n"); log.flush()
model.save_pretrained(args.out)
meta = {"objective": args.objective, "n_traj": len(trajs), "n_pairs": len(pairs) if use_pairs else 0, "steps": total,
        "seconds": round(time.time() - t0, 1), "tokens_processed": n_tok, "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "lr": args.lr, "beta": args.beta, "lam": args.lam, "seed": args.seed}
json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
print(json.dumps(meta))
