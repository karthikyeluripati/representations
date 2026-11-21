"""Evaluation entry point.

Modes
  scripted       CPU. Runs the full suite with scripted policies (oracle / surrogate) through the SAME text-parsing agent
                 loop and graders. Expected: oracle ~1.0 on every suite and false_outage_rate 0; surrogate 0 on RP/RA
                 and false_outage_rate 1. Any deviation is a grader/environment bug.
  run            GPU. vLLM generation for a base or merged model, optionally with a LoRA adapter. Greedy decoding.
  sample_states  GPU. For each TRAIN corrective state, samples K actions from the policy and K continuations per
                 branch (a+ and each sampled failing action) -> policy-verified pairs (CSO-style verification).
  summarize      Metrics (+ cluster-bootstrap CIs) for one or more result files; paired differences vs --ref.

Outputs go to results/eval/<name>.jsonl (+ .summary.json). Raw completions are kept.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.environment.bfcl_env import load_bfcl, openai_tools  # noqa: E402
from src.evaluation import agent_loop as AL  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation.suites import build_specs, load_eval_states  # noqa: E402

PAIRS = os.path.join(ROOT, "data", "cacrt_pairs_v1")
OUT = os.path.join(ROOT, "results", "eval")
HEADLINE = ["fault_free", "genuine_repair", "injected_recovery", "conditional_recovery", "correct_escalation",
            "premature_abstention", "false_outage_rate", "overall"]


def context():
    tasks = json.load(open(os.path.join(ROOT, "data", "tasks_v1.json")))
    by_id = {t["id"]: t for t in tasks}
    _, _, docs = load_bfcl()
    cache = {}

    def tools_fn(tid, alt):
        if (tid, alt) not in cache:
            cache[(tid, alt)] = openai_tools(by_id[tid], docs, AL.ALTS[alt])
        return cache[(tid, alt)]
    return tasks, by_id, tools_fn


def specs_for(tasks, splits):
    st = load_eval_states([os.path.join(PAIRS, f"eval_states_{s}.jsonl") for s in splits])
    return build_specs(tasks, st, splits)


def write(rows, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return p


def summary_with_ci(rows):
    s = M.summarize(rows)
    s["ci"] = {}
    for k in HEADLINE:
        if s.get(k) is not None:
            s["ci"][k] = M.cluster_boot(rows, M.metric_fn(k), B=1000)
    return s


def by_split(rows):
    return {sp: summary_with_ci([r for r in rows if r["split"] == sp]) for sp in sorted({r["split"] for r in rows})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["scripted", "run", "sample_states", "summarize"])
    ap.add_argument("--policy", default="oracle", help="scripted: oracle|surrogate")
    ap.add_argument("--splits", default="test,heldout")
    ap.add_argument("--model", help="run/sample_states: HF path of base or merged model")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--ref", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=16384)
    a = ap.parse_args()
    tasks, by_id, tools_fn = context()
    splits = a.splits.split(",")

    if a.mode == "scripted":
        specs = specs_for(tasks, splits)
        rows = AL.run(specs, AL.ScriptedPolicy(a.policy, by_id), by_id, tools_fn)
        name = a.name or f"scripted_{a.policy}"
        write(rows, name)
        s = by_split(rows)
        json.dump(s, open(os.path.join(OUT, name + ".summary.json"), "w"), indent=1)
        for sp, v in s.items():
            print(sp, json.dumps({k: (round(v[k], 3) if isinstance(v[k], float) else v[k]) for k in HEADLINE + ["n", "n_real_error_events"]}))
            print("   per-suite:", {k: round(x, 3) for k, x in v["success"].items()})
        return

    if a.mode == "summarize":
        res = {}
        ref = [json.loads(l) for l in open(a.ref)] if a.ref else None
        for f in a.files:
            rows = [json.loads(l) for l in open(f)]
            res[f] = by_split(rows)
            if ref:
                res[f]["paired_vs_ref"] = {}
                for sp in sorted({r["split"] for r in rows}):
                    A = [r for r in rows if r["split"] == sp]; B = [r for r in ref if r["split"] == sp]
                    res[f]["paired_vs_ref"][sp] = {k: M.paired_diff(A, B, M.metric_fn(k), B=1000) for k in HEADLINE
                                                   if M.summarize(A).get(k) is not None and M.summarize(B).get(k) is not None}
                    res[f]["paired_vs_ref"][sp]["retention_S0"] = M.retention(A, B)
        print(json.dumps(res, indent=1)[:20000])
        return

    # ---- GPU modes ----
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, enable_lora=bool(a.adapter), max_lora_rank=64, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    lora = LoRARequest("arm", 1, a.adapter) if a.adapter else None
    if a.mode == "run":
        specs = specs_for(tasks, splits)
        pol = AL.VLLMPolicy(llm, tok, SamplingParams(temperature=0.0, max_tokens=512), lora)
        rows = AL.run(specs, pol, by_id, tools_fn)
        name = a.name or os.path.basename(a.adapter or a.model)
        write(rows, name)
        s = by_split(rows)
        json.dump(s, open(os.path.join(OUT, name + ".summary.json"), "w"), indent=1)
        print(json.dumps({sp: {k: v[k] for k in HEADLINE} for sp, v in s.items()}, indent=1))
        return
    if a.mode == "sample_states":
        from src.counterfactuals.policy_sampling import sample_and_verify
        sample_and_verify(llm, tok, lora, by_id, tools_fn, a.k)


if __name__ == "__main__":
    main()
