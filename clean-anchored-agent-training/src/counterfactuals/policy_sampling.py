"""GPU stage of Component A: policy-sampled negatives and policy-continuation verification (CSO-style).

For each TRAIN corrective state s (forced prefix ending in a real error):
  1. sample K first actions from pi0 at s (temperature 1.0);
  2. every distinct sampled action a != a+ becomes a candidate negative;
  3. execute a+ and each candidate from the SAME reconstructed state, then let pi0 continue (temperature 0.7) K times;
  4. retain (s, a+, a-) iff verify_stochastic(succ(a+), succ(a-)) has a positive lower bound.
For each twin anchor, pi0's greedy action is compared with the anchor action: only anchors pi0 already gets right are
"already-correct behaviour" (the rest are reported, and excluded with --policy-correct-anchors).
Costs (env executions, model calls, tokens) are accumulated and written to the manifest.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os

from src.counterfactuals.construct import verify_stochastic
from src.evaluation import agent_loop as AL
from src.replay.replay import call_to_str

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAIRS = os.path.join(ROOT, "data", "cacrt_pairs_v1")
OUT = os.path.join(ROOT, "data", "cacrt_pairs_v1_policy")


def _first_call(text):
    c = AL.parse_tool_calls(text)
    return c[0] if c else None


def sample_and_verify(llm, tok, lora, by_id, tools_fn, k=4):
    from vllm import SamplingParams
    states = [json.loads(l) for l in open(os.path.join(PAIRS, "train_states.jsonl"))]
    states = [s for s in states if s["naive_fails"] and s["outcomes"]["repairer"]["pos"]]
    print(f"{len(states)} train corrective states (valid real-error states whose repair passes the grader)", flush=True)
    base = {"pool": "v2", "alt": "train", "scenario": "S0", "seed": 0, "suite": "branch"}
    first = AL.VLLMPolicy(llm, tok, SamplingParams(temperature=1.0, max_tokens=256, n=1, seed=0), lora)
    cont = AL.VLLMPolicy(llm, tok, SamplingParams(temperature=0.7, max_tokens=512, seed=1), lora)
    cost = collections.Counter()
    # 1-2: sample first actions
    specs = [{**base, "task_id": s["task_id"], "forced": s["forced"], "i": i} for i, s in enumerate(states) for _ in range(k)]
    st = [{"spec": sp, "ep": AL.make_episode(sp, by_id)} for sp in specs]
    from src.counterfactuals.construct import episode_messages
    for x in st:
        x["msgs"] = episode_messages(by_id[x["spec"]["task_id"]], x["ep"]); x["tools"] = tools_fn(x["spec"]["task_id"], "train")
    texts = first.act(st)
    cost["model_calls"] += len(st)
    cand = collections.defaultdict(set)
    for x, t in zip(st, texts):
        c = _first_call(t)
        s = states[x["spec"]["i"]]
        pos = (s["pos_action"]["name"], s["pos_action"]["arguments"])
        if c is None:
            cand[x["spec"]["i"]].add(("__text__", t.strip()[:200]))
        elif c != pos:
            cand[x["spec"]["i"]].add((c[0], json.dumps(c[1], sort_keys=True)))
    # 3: branch + continue
    branch_specs = []
    for i, s in enumerate(states):
        acts = [("pos", s["pos_action"]["name"], json.dumps(s["pos_action"]["arguments"], sort_keys=True))]
        sampled = {(n, a) for n, a in cand[i] if n != "__text__"}
        # CPU-stage candidate negatives (retry / escalate / alternative) are RE-VERIFIED here under pi0 continuations
        for nm, na in s["neg_actions"].items():
            key = (na["name"], json.dumps(na["arguments"], sort_keys=True))
            acts.append((f"cpu_{nm}" + ("+sampled" if key in sampled else ""), key[0], key[1]))
            sampled.discard(key)
        acts += [(f"sampled{j}", n, a) for j, (n, a) in enumerate(sorted(sampled))]
        for tag, n, a in acts:
            forced = s["forced"] + [call_to_str(n, json.loads(a))]
            for r in range(k):
                branch_specs.append({**base, "task_id": s["task_id"], "forced": forced, "i": i, "tag": tag, "rep": r,
                                     "action": {"name": n, "arguments": json.loads(a)}})
    rows = AL.run(branch_specs, cont, by_id, tools_fn)
    cost["model_calls"] += sum(r["n_model_calls"] for r in rows)
    cost["env_executions"] += sum(len(r["calls"]) for r in rows)
    cost["tokens_in"] += sum(r["tok_in"] for r in rows); cost["tokens_out"] += sum(r["tok_out"] for r in rows)
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    acts = {}
    for sp, r in zip(branch_specs, rows):
        res[sp["i"]][sp["tag"]].append(r["correct"]); acts[(sp["i"], sp["tag"])] = sp["action"]
    pairs, stats = [], collections.Counter()
    stats["first_action_samples"] = len(st)
    stats["first_action_equals_repair"] = sum(1 for x, t in zip(st, texts) if _first_call(t) == (states[x["spec"]["i"]]["pos_action"]["name"], states[x["spec"]["i"]]["pos_action"]["arguments"]))
    for i, d in res.items():
        s = states[i]
        stats["states"] += 1
        stats["pos_success_sum"] += sum(d["pos"]); stats["pos_runs"] += len(d["pos"])
        stats["states_pos_never_succeeds"] += (sum(d["pos"]) == 0)
        for tag, succ in d.items():
            if tag == "pos":
                continue
            ok, diff = verify_stochastic(d["pos"], succ)
            src = tag.split("+")[0] if tag.startswith("cpu_") else "sampled"
            stats[f"candidates:{src}"] += 1
            stats[f"{'accepted' if ok else 'rejected'}:{src}"] += 1
            if ok:
                pairs.append({"type": "pair", "task_id": s["task_id"], "msgs": s["messages"], "chosen": s["pos_action"],
                              "rejected": acts[(i, tag)], "neg": tag, "state": s["key"], "succ_pos": d["pos"],
                              "succ_neg": succ, "diff": diff})
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "train_policy_pairs.jsonl")
    with open(p, "w") as f:
        for x in pairs:
            f.write(json.dumps(x) + "\n")
    # policy-correct twin anchors
    tw = [json.loads(l) for l in open(os.path.join(PAIRS, "train_anchors_twin.jsonl"))]
    greedy = AL.VLLMPolicy(llm, tok, SamplingParams(temperature=0.0, max_tokens=256), lora)
    xs = [{"msgs": a["messages"], "tools": tools_fn(a["task_id"], "train")} for a in tw]
    outs = greedy.act(xs)
    keep = []
    for a, t in zip(tw, outs):
        c = _first_call(t)
        want = (a["action"]["name"], a["action"]["arguments"]) if "name" in a["action"] else None
        correct = (c == want) if want else (c is None)
        keep.append({**a, "policy_correct": bool(correct)})
    with open(os.path.join(OUT, "train_anchors_twin_policy.jsonl"), "w") as f:
        for a in keep:
            f.write(json.dumps(a) + "\n")
    stats["twin_policy_correct"] = sum(a["policy_correct"] for a in keep); stats["twin_total"] = len(keep)
    man = {"stats": dict(stats), "cost": dict(cost), "k": k,
           "files": {fn: hashlib.sha256(open(os.path.join(OUT, fn), "rb").read()).hexdigest() for fn in os.listdir(OUT) if fn.endswith(".jsonl")}}
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))
