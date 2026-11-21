"""Chat rendering and loss masks.

Real models: `HFRenderer` uses the tokenizer's own chat template (Qwen3-Instruct-2507 renders tools in the system
prompt and tool calls as <tool_call>{json}</tool_call>), so training and evaluation prompts are identical by
construction. Target tokens of an assistant message are found by prefix rendering:
    prompt = render(msgs[:i], add_generation_prompt=True);  full = render(msgs[:i+1])
    target tokens = full_ids[len(prompt_ids):]      (asserted to be a strict extension)
Tests: `ToyRenderer` (byte-level, no download) with the same interface.
"""
from __future__ import annotations

import json


class HFRenderer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def render(self, msgs, tools, add_generation_prompt=False):
        return self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]


class ToyTokenizer:
    """Byte-level tokenizer: ids 0..255 = bytes, 256 = pad, 257 = eos."""
    pad_token_id, eos_token_id, vocab_size = 256, 257, 258

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class ToyRenderer:
    def __init__(self):
        self.tok = ToyTokenizer()
        self.pad_id = self.tok.pad_token_id

    def render(self, msgs, tools, add_generation_prompt=False):
        s = "T:" + ",".join(t["function"]["name"] for t in (tools or [])) + "\n"
        for m in msgs:
            body = m.get("content") or ""
            for tc in m.get("tool_calls", []) or []:
                body += "<tool_call>" + json.dumps({"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}, sort_keys=True) + "</tool_call>"
            s += f"<{m['role'][0]}>{body}</e>\n"
        if add_generation_prompt:
            s += "<a>"
        return s

    def encode(self, text):
        return self.tok.encode(text)


def action_message(action):
    if "final_text" in action:
        return {"role": "assistant", "content": action["final_text"]}
    return {"role": "assistant", "content": action.get("content", ""),     # optional short rationale before the call
            "tool_calls": [{"type": "function", "function": {"name": action["name"], "arguments": action["arguments"]}}]}


def target_span(renderer, msgs, i, tools):
    """Token ids of the prompt up to message i and of message i itself."""
    p = renderer.encode(renderer.render(msgs[:i], tools, add_generation_prompt=True))
    f = renderer.encode(renderer.render(msgs[:i + 1], tools))
    if f[:len(p)] != p:
        raise ValueError("chat template is not prefix-stable at this message")
    return p, f[len(p):]


def encode_action_example(renderer, msgs, action, tools, max_len):
    """prefix messages + one target action -> (input_ids, loss_mask). None if longer than max_len."""
    full = list(msgs) + [action_message(action)]
    p, t = target_span(renderer, full, len(full) - 1, tools)
    ids = p + t
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "mask": [0] * len(p) + [1] * len(t), "n_target": len(t)}


def encode_trajectory(renderer, msgs, tools, max_len):
    """Whole-trajectory SFT: loss on every assistant message whose `loss` flag is not False."""
    ids, mask = [], []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        p, t = target_span(renderer, msgs, i, tools)
        if ids and p[:len(ids)] != ids:
            raise ValueError("chat template is not prefix-stable across turns")
        mask += [0] * (len(p) - len(ids)) + [int(m.get("loss", True))] * len(t)
        ids = p + t
    if len(ids) > max_len or not ids:
        return None
    return {"input_ids": ids, "mask": mask, "n_target": sum(mask)}
