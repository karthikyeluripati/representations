"""Collect rollouts. Example:
python scripts/collect_rollouts.py --model $M --variant B0 --splits dev --scenarios S0 S1 S2 S3 --seeds 0 --out results/b0_dev.jsonl
Options: --lora <adapter_dir> (vLLM LoRA), --temperature, --n (samples per item; seeds expand items)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from environments.bfcl_faults import load_bfcl
from agents.rollout import Runner, SYSTEM_BASE, SYSTEM_RECOVERY

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--lora", default=None)
ap.add_argument("--variant", choices=["B0", "B1", "B3"], default="B0")
ap.add_argument("--splits", nargs="+", default=["dev"]); ap.add_argument("--scenarios", nargs="+", default=["S0", "S1", "S2", "S3"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0]); ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--out", required=True)
ap.add_argument("--tasks", default="data/tasks.json"); ap.add_argument("--rule-n", type=int, default=4)
args = ap.parse_args()

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

tasks = [t for t in json.load(open(args.tasks)) if t["split"] in args.splits]
_, _, docs = load_bfcl()
tok = AutoTokenizer.from_pretrained(args.model)
llm = LLM(args.model, max_model_len=16384, gpu_memory_utilization=0.85, enable_lora=bool(args.lora), max_lora_rank=64, seed=0)
system = SYSTEM_RECOVERY if args.variant == "B3" else SYSTEM_BASE
runner = Runner(llm, tok, docs, system=system, rule_max_consecutive_errors=args.rule_n if args.variant == "B1" else None)
if args.lora:
    lr = LoRARequest("adapter", 1, args.lora)
    _gen = llm.generate
    llm.generate = lambda prompts, sp, use_tqdm=False: _gen(prompts, sp, use_tqdm=use_tqdm, lora_request=lr)
sp = SamplingParams(temperature=args.temperature, top_p=1.0 if args.temperature == 0 else 0.95, max_tokens=args.max_tokens, seed=0)
items = [(t, sc, s) for t in tasks for sc in args.scenarios for s in args.seeds]
t0 = time.time()
res = runner.run(items, sp)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out + ".tmp", "w") as f:
    for r in res:
        f.write(json.dumps(r, default=str) + "\n")
os.replace(args.out + ".tmp", args.out)
meta = {"model": args.model, "lora": args.lora, "variant": args.variant, "splits": args.splits, "scenarios": args.scenarios, "seeds": args.seeds,
        "temperature": args.temperature, "n_items": len(items), "seconds": round(time.time() - t0, 1),
        "tok_in": sum(r["tok_in"] for r in res), "tok_out": sum(r["tok_out"] for r in res)}
json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
print(json.dumps(meta))
