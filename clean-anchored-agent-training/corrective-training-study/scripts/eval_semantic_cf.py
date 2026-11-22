"""Model evaluation on the semantic-counterfactual diagnostic (docs/semantic_counterfactual_protocol.md). GPU (vLLM).

NOT RUN YET: requires explicit authorization (pod spend). Same loop semantics as eval_tau2_ac.py: greedy, the loop ends
on a reply without a tool call, an escalation or the call cap. Base model + LoRA adapters, one output file per run.
  python scripts/eval_semantic_cf.py --model /workspace/models/piB --runs piB= piB_prompt= SYN64_s0=/path/adapter ... --out DIR
(a run whose name ends in _prompt gets SYSTEM + PROMPT_HINT)
The batched loop (run_batch) is CPU-tested with a text-emitting scripted policy (tests/test_semantic_cf_eval.py).
"""
import argparse
import json
import os
import re

import _common  # noqa: F401  (sys.path)
from verification.semantic_cf import PROMPT_HINT, SYSTEM, CFEpisode, metrics, specs

TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)      # identical to src/evaluation/agent_loop.TC_RE
MAX_STEPS = 10


def parse(text):
    out = []
    for m in TC_RE.finditer(text):
        try:
            d = json.loads(m.group(1)); out.append((d["name"], d.get("arguments", {}) or {}))
        except Exception:  # noqa: BLE001
            out.append(("__malformed__", {}))
    return out


def run_batch(policy, sp, system=SYSTEM):
    """policy.act(states) -> list of raw completions (states carry 'msgs' and 'tools')."""
    st = []
    for s in sp:
        ep = CFEpisode(s); st.append({"spec": s, "ep": ep, "msgs": ep.messages(system), "tools": ep.tools(), "done": False, "steps": []})
    for _ in range(MAX_STEPS):
        act = [s for s in st if not s["done"]]
        if not act:
            break
        for s, text in zip(act, policy.act(act)):
            s["steps"].append(text[:2000]); calls = parse(text); content = TC_RE.sub("", text).strip()
            if not calls:
                s["ep"].finish(content); s["done"] = True; continue
            s["msgs"].append({"role": "assistant", "content": content,
                              "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]})
            for n, a in calls:
                o = s["ep"].call(n, a) if n != "__malformed__" else "Error: malformed tool call"
                s["msgs"].append({"role": "tool", "content": o})
                if s["ep"].done:
                    break
            s["done"] = s["ep"].done
    return [{**s["spec"], **s["ep"].grade(), "calls": [[n, a, o[:200]] for n, a, o in s["ep"].calls],
             "final_text": s["ep"].final_text, "completions": s["steps"]} for s in st]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    sp = specs(a.split)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import sys
    sys.path.insert(0, _common.PROJ)
    from src.evaluation.agent_loop import VLLMPolicy
    runs = [x.split("=", 1) for x in a.runs]
    llm = LLM(model=a.model, enable_lora=any(d for _, d in runs), max_lora_rank=64, max_model_len=16384, gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer(); os.makedirs(a.out, exist_ok=True)
    for k, (name, d) in enumerate(runs):
        p = os.path.join(a.out, name + ".jsonl")
        if os.path.exists(p):
            continue
        system = SYSTEM + " " + PROMPT_HINT if name.endswith("_prompt") else SYSTEM      # prompt-only arm: name it e.g. piB_prompt
        rows = run_batch(VLLMPolicy(llm, tok, SamplingParams(temperature=0.0, max_tokens=512), LoRARequest(name, k + 1, d) if d else None), sp, system)
        with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(p + ".tmp", p)
        m = metrics(rows); json.dump({k2: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k2, v in m.items()},
                                     open(os.path.join(a.out, name + ".summary.json"), "w"), indent=1)
        print(name, flush=True)


if __name__ == "__main__":
    main()
