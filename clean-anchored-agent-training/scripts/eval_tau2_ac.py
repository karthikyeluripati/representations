"""Zero-shot transfer: uncertain-outcome faults on tau2-bench airline (pre-registered in
corrective-training-study/docs/tau2_transfer.md). No tau2 data is used in any training.

Episodes per policy:
  diagnostic events (non-idempotent, readable writes): S0 + {ACa, ACl} x {explicit, para, implicit}
  control events (idempotent writes):                  S0 + {ACa, ACl} x {implicit}
Same agent loop semantics as BFCL: greedy, the loop ends on a reply without a tool call, an escalation or the call cap.
  python scripts/eval_tau2_ac.py --model PIB --runs name=adapter ... --out DIR
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from src.counterfactuals.construct import SYSTEM  # noqa: E402
from src.environment.bfcl_env import AC_MESSAGES  # noqa: E402
from src.environment.tau2_airline import Tau2Episode, _load, events, tools  # noqa: E402
from src.evaluation.agent_loop import TC_RE, VLLMPolicy, parse_tool_calls  # noqa: E402

MAX_STEPS = 20
MAX_LEN = 40960; OBS_CHARS = 2000          # same observation truncation as the BFCL loop


def specs():
    E = events(); out = []
    for kind, msgs in (("diagnostic", ("explicit", "para", "implicit")), ("control", ("implicit",))):
        for tid, i, fn in E[kind]:
            out.append({"kind": kind, "task": tid, "i": i, "fn": fn, "scenario": "S0", "msg": None})
            for m in msgs:
                for sc in ("ACa", "ACl"):
                    out.append({"kind": kind, "task": tid, "i": i, "fn": fn, "scenario": sc, "msg": m})
    return out


def run(sp, policy):
    states = []
    for s in sp:
        ep = Tau2Episode(s["task"], s["i"], s["scenario"], AC_MESSAGES[s["msg"]] if s["msg"] else "")
        states.append({"spec": s, "ep": ep, "msgs": ep.messages(SYSTEM), "tools": tools(ep.env), "done": False, "steps": []})
    for _ in range(MAX_STEPS):
        for s in states:                     # end (do not crash on) episodes whose context would overflow
            if not s["done"] and len(policy.tok.apply_chat_template(s["msgs"], tools=s["tools"], add_generation_prompt=True)) > MAX_LEN - 600:
                s["done"] = True; s["ep"].finish(); s["steps"].append("[context limit]")
        act = [s for s in states if not s["done"]]
        if not act:
            break
        for s, text in zip(act, policy.act(act)):
            calls = parse_tool_calls(text); s["steps"].append(text[:3000])
            content = TC_RE.sub("", text).strip()
            if not calls:
                s["msgs"].append({"role": "assistant", "content": content}); s["done"] = True; continue
            s["msgs"].append({"role": "assistant", "content": content,
                              "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]})
            for n, a in calls:
                o = s["ep"].call(n, a) if n != "__malformed__" else "Error: malformed tool call"
                s["msgs"].append({"role": "tool", "content": o[:OBS_CHARS]})
                if s["ep"].done:
                    break
            if s["ep"].done:
                s["done"] = True
    rows = []
    for s in states:
        g = s["ep"].grade()
        rows.append({**s["spec"], **g, "calls": [(n, a, o[:300]) for n, a, o in s["ep"].calls], "n_forced": s["ep"].n_forced,
                     "completions": s["steps"]})
    return rows


def after_error(r):
    c = r["calls"]; k = r["n_forced"]
    if r["scenario"] == "S0" or k >= len(c):
        return "stopped" if r["scenario"] != "S0" else None
    nx = c[k][0]; tgt = c[k - 1][0]
    return "retry" if nx == tgt else "escalate" if nx in ("escalate_to_user", "transfer_to_human_agents") else \
        "check" if nx.startswith("get_") else "other"


def summarize(rows):
    out = {}
    for kind in ("diagnostic", "control"):
        rs = [r for r in rows if r["kind"] == kind]
        s0 = {(r["task"], r["i"]): r["correct"] for r in rs if r["scenario"] == "S0"}
        out[kind] = {"S0": sum(s0.values()) / max(len(s0), 1)}
        for m in ("explicit", "para", "implicit"):
            sub = [r for r in rs if r["msg"] == m]
            if not sub:
                continue
            cond = [r["correct"] for r in sub if s0.get((r["task"], r["i"]))]
            out[kind][m] = {"success": sum(r["correct"] for r in sub) / len(sub), "success_given_S0": sum(cond) / max(len(cond), 1),
                            "n": len(sub), "after_error": dict(collections.Counter(after_error(r) for r in sub)),
                            "mean_calls": sum(r["n_calls"] for r in sub) / len(sub)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _load()
    sp = specs()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    runs = [x.split("=", 1) for x in a.runs]
    llm = LLM(model=a.model, enable_lora=any(d for _, d in runs), max_lora_rank=64, max_model_len=MAX_LEN, gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    os.makedirs(a.out, exist_ok=True)
    for k, (name, d) in enumerate(runs):
        p = os.path.join(a.out, name + ".jsonl")
        if os.path.exists(p):
            continue
        lora = LoRARequest(name, k + 1, d) if d else None
        rows = run(sp, VLLMPolicy(llm, tok, SamplingParams(temperature=0.0, max_tokens=512), lora))
        with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(p + ".tmp", p)
        s = summarize(rows); json.dump(s, open(os.path.join(a.out, name + ".summary.json"), "w"), indent=1)
        print(name, json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
