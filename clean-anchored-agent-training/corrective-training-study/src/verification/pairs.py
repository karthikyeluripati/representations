"""Verified corrective pairs and matched balanced-SFT decisions, rendered for TRL.

Text format (TRL DPOTrainer / SFTTrainer "prompt-completion" style):
  prompt     = tokenizer.apply_chat_template(state_messages, tools, add_generation_prompt=True, tokenize=False)
  completion = the assistant message body exactly as the chat template renders it, WITHOUT the closing <|im_end|>\n
               (TRL appends tokenizer.eos_token_id == <|im_end|> for Qwen3-Instruct-2507).
`check_split_tokenization` asserts tokens(prompt) + tokens(completion) + [eos] == template tokens of the full
conversation up to and including <|im_end|>, i.e. TRL's separate tokenization equals the model's real input.
"""
from __future__ import annotations

import json
import random

from src.training.render import action_message


def render_pair_texts(tok, msgs, tools, action):
    prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(list(msgs) + [action_message(action)], tools=tools, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("template not prefix-stable")
    body = full[len(prompt):]
    end = tok.eos_token + "\n"
    if not body.endswith(end):
        raise ValueError(f"unexpected assistant terminator: {body[-30:]!r}")
    return prompt, body[: -len(end)]


def check_split_tokenization(tok, msgs, tools, action):
    p, c = render_pair_texts(tok, msgs, tools, action)
    split = tok(p, add_special_tokens=False)["input_ids"] + tok(c, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    full = tok.apply_chat_template(list(msgs) + [action_message(action)], tools=tools, tokenize=False)
    joint = tok(full, add_special_tokens=False)["input_ids"]
    return split == joint[: len(split)] and joint[len(split):] == tok("\n", add_special_tokens=False)["input_ids"]


def dpo_records(tok, pairs, tools_fn):
    out = []
    for p in pairs:
        tools = tools_fn(p["task_id"])
        prompt, ch = render_pair_texts(tok, p["msgs"], tools, p["chosen"])
        _, rj = render_pair_texts(tok, p["msgs"], tools, p["rejected"])
        out.append({"prompt": prompt, "chosen": ch, "rejected": rj, "state": p["state"], "neg": p["neg"], "task_id": p["task_id"]})
    return out


def decision_records_from_episode(task, ep, history_fn, include_final=True):
    """Every agent decision of an executed reference episode -> (prefix messages, action)."""
    msgs = history_fn(task)
    out = []
    for n, a, o in ep.calls:
        out.append((list(msgs), {"name": n, "arguments": a}))
        msgs = msgs + [action_message({"name": n, "arguments": a}),
                       {"role": "tool", "content": ("Escalated to the user." if o == "ESCALATED" else o)[:2000]}]
    if include_final and not ep.escalated:
        out.append((list(msgs), {"final_text": "Done."}))
    return out


def balanced_sft_decisions(train_tasks, pairs, n_per_category, seed=0):
    """Three equal categories of verified-correct decisions (all from TRAIN tasks; executed by the reference policy):
      repair     chosen action a+ at each verified corrective state (one per state; real BFCL errors)
      outage     the agent decision immediately after an injected outage in S1/S2/S3 reference episodes
      fault_free decisions of S0 reference episodes (next ground-truth call, or the final reply)
    Each record: {"task_id", "msgs", "action", "category"}."""
    from src.counterfactuals.construct import history_messages
    from src.counterfactuals.policies import reference_policy
    from src.environment.bfcl_env import TRAIN_ERRORS, Episode, obs_kind
    rng = random.Random(seed)
    repair, seen = [], set()
    for p in pairs:
        if p["state"] not in seen:
            seen.add(p["state"])
            repair.append({"task_id": p["task_id"], "msgs": p["msgs"], "action": p["chosen"], "category": "repair"})
    outage, clean = [], []
    for t in train_tasks:
        ep = Episode(t, "S0", 0, error_pool=TRAIN_ERRORS); reference_policy(ep)
        clean += [{"task_id": t["id"], "msgs": m, "action": a, "category": "fault_free"} for m, a in decision_records_from_episode(t, ep, history_messages)]
        for sc in ("S1", "S2", "S3"):
            ep = Episode(t, sc, 0, error_pool=TRAIN_ERRORS); reference_policy(ep)
            dec = decision_records_from_episode(t, ep, history_messages)
            for i, (m, a) in enumerate(dec):
                if m[-1]["role"] == "tool" and obs_kind(m[-1]["content"]) == "injected":
                    outage.append({"task_id": t["id"], "msgs": m, "action": a, "category": "outage", "scenario": sc})
    groups = {"repair": repair, "outage": outage, "fault_free": clean}
    avail = {k: len(v) for k, v in groups.items()}
    picked = [x for k, v in groups.items() for x in rng.sample(v, min(n_per_category, len(v)))]
    rng.shuffle(picked)
    return picked, avail


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]
