"""Uncertain-outcome study (pre-registered in corrective-training-study/docs/uncertain_outcome_study.md).

Per policy, the 116 diagnostic tasks x 2 prompts x 8 suites = 1,856 greedy episodes, run as ONE batch:
  prompts : std   = the system prompt used for all training/evaluation so far
            check = std + one sentence telling the agent to verify state when an outcome is unclear
  suites  : S0 (fault-free), S1 (transient outage control: retrying is correct),
            ACa/ACl x message in {explicit, para (unseen paraphrase), implicit (plain timeout)}
  python scripts/eval_uncertain_outcome.py --model M [--runs name=adapter ...] [--no-thinking] --out DIR
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from evaluate import context  # noqa: E402
from src.counterfactuals.construct import SYSTEM  # noqa: E402
from src.environment.bfcl_env import AC_MESSAGES, ESCALATE  # noqa: E402
from src.evaluation import agent_loop as AL  # noqa: E402

CHECK = SYSTEM + (" If a failure leaves it unclear whether an operation already took effect, do not simply repeat it: first "
                  "check the current state with a read-only tool, and repeat the operation only if it has not taken effect.")
PROMPTS = {"std": SYSTEM, "check": CHECK}
SUITES = [("S0", "S0", None), ("S1", "S1", None)] + [(f"{v}_{m}", v, m) for m in ("explicit", "para", "implicit") for v in ("ACa", "ACl")]
READS = ("get", "list", "display", "view", "check", "ls", "cat", "find", "show", "search", "pwd", "du", "wc", "tail", "head",
         "grep", "sort", "diff", "estimate", "retrieve", "fetch", "read", "verify", "query", "lookup")
AC_OBS = set(AC_MESSAGES.values())


def specs(rows, prompts=tuple(PROMPTS)):
    out = []
    for r in rows:
        if not r["diagnostic"]:
            continue
        for pn, ps in ((k, PROMPTS[k]) for k in prompts):
            for suite, sc, msg in SUITES:
                s = {"task_id": r["task_id"], "pool": "v2", "alt": "train", "suite": suite, "scenario": sc, "seed": 0,
                     "prompt": pn, "system": ps}
                if msg:
                    s["ac_msg"] = msg
                out.append(s)
    return out


def after_error(r):
    c = r["calls"]
    for i, (n, a, o) in enumerate(c):
        if o in AC_OBS:
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
    res = {}
    for pn in sorted({r["prompt"] for r in rows}):
        rp = [r for r in rows if r["prompt"] == pn]
        s0 = {r["task_id"]: r["correct"] for r in rp if r["suite"] == "S0"}
        res[pn] = {}
        for suite, _, msg in SUITES:
            rs = [r for r in rp if r["suite"] == suite]
            cond = [r["correct"] for r in rs if s0.get(r["task_id"])]
            res[pn][suite] = {"n": len(rs), "success": sum(r["correct"] for r in rs) / max(len(rs), 1),
                              "success_given_S0": sum(cond) / max(len(cond), 1), "n_S0_ok": len(cond),
                              "escalated": sum(r["escalated"] for r in rs),
                              "after_error": dict(collections.Counter(after_error(r) for r in rs)) if msg else None}
        for m in ("explicit", "para", "implicit"):
            res[pn][f"AC_{m}_given_S0"] = (res[pn][f"ACa_{m}"]["success_given_S0"] + res[pn][f"ACl_{m}"]["success_given_S0"]) / 2
    return res


def headline(s):
    return {pn: {"S0": round(s[pn]["S0"]["success"], 3), "S1|S0": round(s[pn]["S1"]["success_given_S0"], 3),
                 **{m: round(s[pn][f"AC_{m}_given_S0"], 3) for m in ("explicit", "para", "implicit")}} for pn in s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=adapter_dir (empty = no adapter)")
    ap.add_argument("--no-thinking", action="store_true", help="hybrid Qwen3 models: enable_thinking=False")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--feasibility", default=os.path.join(ROOT, "results", "feasibility_ambiguous_commit.json"))
    ap.add_argument("--prompts", default="std,check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [x.split("=", 1) for x in a.runs]
    os.makedirs(a.out, exist_ok=True)
    todo = [(n, d) for n, d in runs if not os.path.exists(os.path.join(a.out, n + ".jsonl"))]
    if not todo:
        return
    tasks, by_id, tools_fn = context()
    sp = specs(json.load(open(a.feasibility, encoding="utf-8")), tuple(a.prompts.split(",")))
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, enable_lora=any(d for _, d in todo), max_lora_rank=64, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    tkw = {"enable_thinking": False} if a.no_thinking else None
    for k, (name, d) in enumerate(todo):
        lora = LoRARequest(name, k + 1, d) if d else None
        rows = AL.run(sp, AL.VLLMPolicy(llm, tok, SamplingParams(temperature=0.0, max_tokens=512), lora, tkw), by_id, tools_fn)
        p = os.path.join(a.out, name + ".jsonl")
        with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                r.pop("system", None)
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(p + ".tmp", p)
        s = summarize(rows)
        json.dump(s, open(os.path.join(a.out, name + ".summary.json"), "w"), indent=1)
        print(name, json.dumps(headline(s)), flush=True)


if __name__ == "__main__":
    main()
