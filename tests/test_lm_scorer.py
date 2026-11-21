"""KV-cached batched candidate scoring must equal naive full-sequence scoring.

Requires the model weights; opt in with DARE_LM_TESTS=1 (slow on CPU: minutes).
"""
import os

import pytest

if os.environ.get("DARE_LM_TESTS") != "1":
    pytest.skip("set DARE_LM_TESTS=1 to run LM tests", allow_module_level=True)

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from dare.agents.lm import LMScorer, build_prompt_messages, static_user_prefix  # noqa: E402

MODEL = os.environ.get("DARE_LM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def test_cached_equals_naive():
    s = LMScorer(MODEL, max_batch=2)  # max_batch=2 also exercises chunking + cache reuse
    cands = ["go to drawer 1", "go to drawer 10", "look", "take mug 1 from countertop 2"]
    msgs = build_prompt_messages("You are in a room.\n\nYour task is to: put a mug in drawer.", [],
                                 "You are in a room.", cands, 8)
    r = s.score(msgs, cands)
    pids = s.prompt_ids(msgs)
    for k, c in enumerate(cands):
        ids = s.tok.encode(c, add_special_tokens=False) + [s.eot_id]
        with torch.no_grad():
            x = torch.tensor([pids + ids], device=s.device)
            lp = torch.log_softmax(s.model(x, logits_to_keep=len(ids) + 1).logits[0].float(), -1).cpu()
        naive = sum(lp[j, ids[j]].item() for j in range(len(ids)))
        assert r.logp_sum[k] == pytest.approx(naive, abs=2e-3), c
    assert r.action_emb.shape == (len(cands), s.hidden_size)


def _episode_prompts(demo="> look\nYou see a mug."):
    intro = "You are in a room.\n\nYour task is to: put a mug in drawer."
    cands = ["go to drawer 1", "look", "take mug 1 from countertop 2"]
    out, hist = [], []
    for step in range(3):
        out.append((build_prompt_messages(intro, hist, f"obs {step}", cands, 8, demo), static_user_prefix(intro, demo)))
        hist.append((cands[step], f"result {step}"))
    return out, cands


def test_static_prefix_reuse_close_to_full_pass_fp32():
    s = LMScorer(MODEL, max_batch=3, reuse_prefix=True)
    prompts, cands = _episode_prompts()
    for msgs, sp in prompts:
        r = s.score(msgs, cands, sp)
        f = s.score(msgs, cands)  # static_prefix=None -> single full pass
        assert r.logp_sum == pytest.approx(f.logp_sum, abs=2e-3)
        assert r.prompt_tokens_computed < r.prompt_tokens == f.prompt_tokens_computed


def test_static_prefix_reuse_is_history_independent():
    """Branching needs scores that depend on the prompt only, not on which calls came before."""
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    s = LMScorer(MODEL, dtype=dtype, max_batch=3, reuse_prefix=True)
    prompts, cands = _episode_prompts()
    other, _ = _episode_prompts(demo="> inventory\nYou carry nothing.")
    a = s.score(prompts[2][0], cands, prompts[2][1])
    for msgs, sp in other + prompts[:2]:
        s.score(msgs, cands, sp)
    b = s.score(prompts[2][0], cands, prompts[2][1])
    assert np.array_equal(a.logp_sum, b.logp_sum) and np.array_equal(a.action_emb, b.action_emb)
