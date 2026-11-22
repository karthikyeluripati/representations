"""Model evaluation for the matched CF-SFT pilot (docs/cf_sft_pilot_protocol.md). GPU (vLLM). NOT RUN: needs authorization.

  python scripts/eval_cf_pilot.py --model /workspace/models/piB --template T2 --split dev --out DIR \
      --runs piB@P0= piB@P2= CAW2_s0@P3=/path/adapter ...
--template T1 evaluates the FROZEN T1 diagnostic (semantic_cf.specs("test"), unchanged); T2 uses semantic_cf_t2.
A run is NAME@PROMPT=ADAPTER (PROMPT in semantic_cf_t2.PROMPTS; empty adapter = base model). Output NAME@PROMPT.jsonl.
"""
import argparse
import json
import os

import _common  # noqa: F401
from eval_semantic_cf import TC_RE, parse
from verification import semantic_cf as T1
from verification import semantic_cf_t2 as T2

MAX_STEPS = 10


def episodes(template, split):
    if template == "T1":
        assert split == "test"
        return T1.specs("test"), T1.CFEpisode, T1.metrics
    return T2.specs(split), T2.T2Episode, T2.metrics


def run_batch(policy, sp, Ep, system):
    st = []
    for s in sp:
        ep = Ep(s); st.append({"spec": s, "ep": ep, "msgs": ep.messages(system), "tools": ep.tools(), "done": False, "steps": []})
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
    ap.add_argument("--model", required=True); ap.add_argument("--template", choices=("T1", "T2"), required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp, Ep, metrics = episodes(a.template, a.split)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import sys
    sys.path.insert(0, _common.PROJ)
    from src.evaluation.agent_loop import VLLMPolicy
    runs = []
    for x in a.runs:
        np_, d = x.split("=", 1); name, pid = np_.split("@"); assert pid in T2.PROMPTS, pid; runs.append((name, pid, d))
    llm = LLM(model=a.model, enable_lora=any(d for *_, d in runs), max_lora_rank=64, max_model_len=16384, gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer(); os.makedirs(a.out, exist_ok=True); lora_ids = {}
    for name, pid, d in runs:
        p = os.path.join(a.out, f"{name}@{pid}.jsonl")
        if os.path.exists(p):
            continue
        lr = LoRARequest(name, lora_ids.setdefault(d, len(lora_ids) + 1), d) if d else None
        rows = run_batch(VLLMPolicy(llm, tok, SamplingParams(temperature=0.0, max_tokens=512), lr), sp, Ep, T2.system_for(pid))
        with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(p + ".tmp", p)
        m = metrics(rows)
        json.dump({k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in m.items()},
                  open(p.replace(".jsonl", ".summary.json"), "w"), indent=1)
        print(name, pid, flush=True)


if __name__ == "__main__":
    main()
