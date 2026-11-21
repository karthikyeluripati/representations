"""Numerical checks of the TRL-based baselines on a tiny random Qwen3 with the REAL Qwen3-4B tokenizer and real
records. Set CTS_TOKENIZER to a directory with tokenizer.json + tokenizer_config.json (skipped otherwise)."""
import math
import os
import sys

import pytest
import torch

STUDY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(STUDY, "src"))
TOK = os.environ.get("CTS_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOK, reason="CTS_TOKENIZER not set")


@pytest.fixture(scope="module")
def ctx():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from src.training.run import load_context
    from verification.pairs import dpo_records, load_jsonl, render_pair_texts
    tok = AutoTokenizer.from_pretrained(TOK)
    tasks, by_id, tools_fn = load_context()
    pairs = load_jsonl(os.path.join(STUDY, "data", "policy_pairs_v1.jsonl"))
    # two short pairs with different negatives
    pairs = sorted(pairs, key=lambda p: len(str(p["msgs"])))[:2]
    recs = dpo_records(tok, pairs, tools_fn)
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=8, tie_word_embeddings=True, max_position_embeddings=32768)
    return tok, Qwen3ForCausalLM(cfg), recs, pairs, tools_fn, render_pair_texts


def _manual_logp(model, tok, prompt, completion):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = torch.tensor([p + c])
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
    return float(sum(lp[len(p) - 1 + i, t] for i, t in enumerate(c)))


def _dpo_trainer(tok, model, recs, out, lr=1e-2, steps=1):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    from training.trl_runner import lora_config
    cfg = DPOConfig(output_dir=out, learning_rate=lr, max_steps=steps, per_device_train_batch_size=2, beta=0.1,
                    loss_type="sigmoid", use_cpu=True, bf16=False, report_to=[], max_length=16384, max_prompt_length=None,
                    max_completion_length=512, use_logits_to_keep=True, logging_steps=1, save_strategy="no",
                    remove_unused_columns=False, lr_scheduler_type="constant", warmup_ratio=0.0)
    ds = Dataset.from_list([{k: x[k] for k in ("prompt", "chosen", "rejected")} for x in recs])
    return DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora_config(8, 16))


def test_trl_logps_match_independent_computation_and_init_loss_is_log2(ctx, tmp_path):
    tok, model, recs, *_ = ctx
    tr = _dpo_trainer(tok, model, recs, str(tmp_path))
    batch = tr.data_collator([tr.train_dataset[i] for i in range(len(recs))])
    with torch.no_grad():
        out = tr.concatenated_forward(tr.model, batch)
    for i, r in enumerate(recs):
        assert abs(float(out["chosen_logps"][i]) - _manual_logp(tr.model, tok, r["prompt"], r["chosen"])) < 1e-3
        assert abs(float(out["rejected_logps"][i]) - _manual_logp(tr.model, tok, r["prompt"], r["rejected"])) < 1e-3
    with torch.no_grad():
        loss, metrics = tr.get_batch_loss_metrics(tr.model, batch, train_eval="train")
    assert math.isclose(float(loss), math.log(2), abs_tol=1e-4)      # LoRA B=0 -> policy == reference (pi0)


def test_trl_dpo_moves_preference_in_the_right_direction(ctx, tmp_path):
    tok, model, recs, *_ = ctx
    import copy
    tr = _dpo_trainer(tok, copy.deepcopy(model), recs, str(tmp_path), lr=5e-2, steps=6)
    tr.train()
    margins = [h["rewards/margins"] for h in tr.state.log_history if "rewards/margins" in h]
    assert margins[-1] > margins[0] and margins[-1] > 0


def test_sft_completion_mask_covers_exactly_completion_plus_eos(ctx, tmp_path):
    tok, model, recs, pairs, tools_fn, render = ctx
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from training.trl_runner import lora_config
    import copy
    pr, co = render(tok, pairs[0]["msgs"], tools_fn(pairs[0]["task_id"]), pairs[0]["chosen"])
    cfg = SFTConfig(output_dir=str(tmp_path), max_steps=1, use_cpu=True, bf16=False, report_to=[], max_length=16384,
                    completion_only_loss=True, save_strategy="no")
    tr = SFTTrainer(model=copy.deepcopy(model), args=cfg, train_dataset=Dataset.from_list([{"prompt": pr, "completion": co}]),
                    processing_class=tok, peft_config=lora_config(8, 16))
    ex = tr.train_dataset[0]
    want = tok(co, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    got = [t for t, m in zip(ex["input_ids"], ex["completion_mask"]) if m]
    assert got == want
    assert ex["input_ids"][: len(ex["input_ids"]) - len(want)] == tok(pr, add_special_tokens=False)["input_ids"]


def test_precomputed_reference_matches_on_the_fly(ctx, tmp_path):
    """precompute_ref_log_probs=True (used for replication) must give the same initial loss (log 2) and train."""
    tok, model, recs, *_ = ctx
    import copy
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    from training.trl_runner import lora_config
    cfg = DPOConfig(output_dir=str(tmp_path), learning_rate=5e-2, max_steps=4, per_device_train_batch_size=2, beta=0.1,
                    loss_type="sigmoid", use_cpu=True, bf16=False, report_to=[], max_length=16384, max_prompt_length=None,
                    max_completion_length=512, use_logits_to_keep=True, logging_steps=1, save_strategy="no",
                    remove_unused_columns=False, lr_scheduler_type="constant", precompute_ref_log_probs=True)
    ds = Dataset.from_list([{k: x[k] for k in ("prompt", "chosen", "rejected")} for x in recs])
    tr = DPOTrainer(model=copy.deepcopy(model), args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora_config(8, 16))
    tr.train()
    h = [x for x in tr.state.log_history if "loss" in x]
    assert math.isclose(h[0]["loss"], math.log(2), abs_tol=1e-3) and h[-1]["rewards/margins"] > 0
