"""Headroom measurement on the ambiguous-commit suite (pre-registered gate in
corrective-training-study/docs/headroom_ambiguous_commit.md). No training, no selection.

Episodes: every diagnostic task (results/feasibility_ambiguous_commit.json) x {S0, ACa, ACl}, greedy, same loop as all
other evaluations. One vLLM engine per base model; adapters are served as LoRA requests.
  python scripts/eval_ambiguous_commit.py --model /workspace/models/pi0 --runs pi0= B_s0=<dir> ... --out results/ac
  python scripts/eval_ambiguous_commit.py --model /root/models/Qwen3-4B-Instruct-2507 --runs base= --out results/ac
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from evaluate import context  # noqa: E402
from src.environment.bfcl_env import AMBIGUOUS_COMMIT, ESCALATE  # noqa: E402
from src.evaluation import agent_loop as AL  # noqa: E402

READS = ("get", "list", "display", "view", "check", "ls", "cat", "find", "show", "search", "pwd", "du", "wc", "tail", "head",
         "grep", "sort", "diff", "estimate", "retrieve", "fetch", "read", "verify", "query", "lookup")


def specs(rows):
    out = []
    for r in rows:
        if r["diagnostic"]:
            for sc in ("S0", "ACa", "ACl"):
                out.append({"task_id": r["task_id"], "pool": "v2", "alt": "train", "suite": sc, "scenario": sc, "seed": 0})
    return out


def after_error(r):
    """What the agent did right after the ambiguous-commit observation."""
    c = r["calls"]
    for i, (n, a, o) in enumerate(c):
        if o.startswith(AMBIGUOUS_COMMIT[:40]):
            if i + 1 == len(c):
                return "stopped"
            nn = c[i + 1][0]
            if nn == ESCALATE:
                return "escalate"
            if nn in (n, n + "_backup"):
                return "retry"
            return "verify" if nn.lstrip("_").lower().startswith(READS) else "other_call"
    return "no_error_seen"


def summarize(rows):
    s0 = {r["task_id"]: r["correct"] for r in rows if r["suite"] == "S0"}
    res = {}
    for sc in ("S0", "ACa", "ACl"):
        rs = [r for r in rows if r["suite"] == sc]
        cond = [r["correct"] for r in rs if s0.get(r["task_id"])]
        res[sc] = {"n": len(rs), "success": sum(r["correct"] for r in rs) / max(len(rs), 1),
                   "success_given_S0": sum(cond) / max(len(cond), 1), "n_S0_ok": len(cond),
                   "after_error": dict(collections.Counter(after_error(r) for r in rs if sc != "S0"))}
    res["AC_mean_given_S0"] = (res["ACa"]["success_given_S0"] + res["ACl"]["success_given_S0"]) / 2
    res["by_split"] = {sp: {sc: sum(r["correct"] for r in rows if r["suite"] == sc and r["split"] == sp) /
                            max(sum(1 for r in rows if r["suite"] == sc and r["split"] == sp), 1) for sc in ("S0", "ACa", "ACl")}
                       for sp in sorted({r["split"] for r in rows})}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=adapter_dir (empty dir = no adapter)")
    ap.add_argument("--feasibility", default=os.path.join(ROOT, "results", "feasibility_ambiguous_commit.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks, by_id, tools_fn = context()
    sp = specs(json.load(open(a.feasibility, encoding="utf-8")))
    runs = [x.split("=", 1) for x in a.runs]
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, enable_lora=any(d for _, d in runs), max_lora_rank=64, max_model_len=16384,
              gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    os.makedirs(a.out, exist_ok=True)
    for k, (name, d) in enumerate(runs):
        p = os.path.join(a.out, name + ".jsonl")
        if os.path.exists(p):
            continue
        lora = LoRARequest(name, k + 1, d) if d else None
        rows = AL.run(sp, AL.VLLMPolicy(llm, tok, SamplingParams(temperature=0.0, max_tokens=512), lora), by_id, tools_fn)
        with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(p + ".tmp", p)
        s = summarize(rows)
        json.dump(s, open(os.path.join(a.out, name + ".summary.json"), "w"), indent=1)
        print(name, json.dumps({k2: s[k2] for k2 in ("S0", "ACa", "ACl", "AC_mean_given_S0")}), flush=True)


if __name__ == "__main__":
    main()
