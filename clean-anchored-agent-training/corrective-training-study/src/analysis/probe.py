"""Error-recognition probe: does the policy RECOGNISE the error type, independently of the action it takes?

For every real-error evaluation state (RP/RA: forced prefix ending in a REAL BFCL error) and its outage twin (same
correct prefix, the same call returns an injected outage drawn from the reserved EVAL pool), the probe appends one
user turn and asks for a single letter:
    "Before you continue: was the last tool result (A) a temporary service outage that may succeed if retried, or
     (B) an error caused by the request itself (wrong argument, missing prerequisite step, or invalid state)?
     Answer with exactly one letter: A or B."
Correct: B for real errors, A for outages. Greedy decoding, max 4 new tokens. The probe never feeds back into the
episode. Combined with the policy's actual next action at the same state (from the main evaluation), this separates
"fails to recognise" (wrong letter) from "recognises but acts wrongly" (right letter, outage response to a real error).
"""
from __future__ import annotations

import json

from src.counterfactuals.construct import Counter, episode_messages, make_episode
from src.environment.bfcl_env import EVAL_ERRORS, obs_kind
from src.replay.replay import parse_call

QUESTION = ("Before you continue: was the last tool result (A) a temporary service outage that may succeed if retried, "
            "or (B) an error caused by the request itself (wrong argument, missing prerequisite step, or invalid state)? "
            "Answer with exactly one letter: A or B.")


def probe_items(eval_states, by_id):
    c = Counter()
    items = []
    for s in eval_states:
        if not s["retained_pairs"]:
            continue
        t = by_id[s["task_id"]]
        ep = make_episode(t, s["forced"], c)
        items.append({"key": s["key"], "split": s["split"], "base_id": t["base_id"], "kind": "real", "want": "B",
                      "msgs": episode_messages(t, ep), "task_id": t["id"]})
        n = parse_call(s["pivot_call"])[0]
        tw = make_episode(t, s["correct_prefix"] + [s["pivot_call"]], c, scenario="S1", inject_fn=n, k=1,
                          inject_from=len(s["correct_prefix"]), error_pool=EVAL_ERRORS)
        if tw.calls and obs_kind(tw.calls[-1][2]) == "injected":
            items.append({"key": s["key"], "split": s["split"], "base_id": t["base_id"], "kind": "outage", "want": "A",
                          "msgs": episode_messages(t, tw), "task_id": t["id"]})
    return items


def run_probe(llm, tok, lora, items, tools_fn):
    from vllm import SamplingParams
    prompts = [tok.apply_chat_template(it["msgs"] + [{"role": "user", "content": QUESTION}], tools=tools_fn(it["task_id"], "train"),
                                       add_generation_prompt=True, tokenize=False) for it in items]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False, lora_request=lora)
    res = []
    for it, o in zip(items, outs):
        txt = o.outputs[0].text.strip().upper()
        ans = "A" if txt.startswith("A") else ("B" if txt.startswith("B") else "?")
        res.append({k: it[k] for k in ("key", "split", "base_id", "kind", "want")} | {"answer": ans, "raw": o.outputs[0].text[:20]})
    return res


def summarize_probe(res):
    out = {}
    for sp in sorted({r["split"] for r in res}):
        for kind in ("real", "outage"):
            R = [r for r in res if r["split"] == sp and r["kind"] == kind]
            if R:
                out[f"{sp}:{kind}"] = {"n": len(R), "correct": sum(r["answer"] == r["want"] for r in R),
                                       "unparsed": sum(r["answer"] == "?" for r in R)}
    return out


def dump(res, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
