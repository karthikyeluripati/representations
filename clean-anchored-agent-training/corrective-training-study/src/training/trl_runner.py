"""Baseline training with TRL 0.24.0 (established framework; no custom optimizer stack).

Baseline B, CSO-style verified corrective DPO: trl.DPOTrainer, loss_type="sigmoid" (standard DPO = CSO Eq. 3),
  reference = the frozen starting policy pi0 (PEFT adapter disabled = pi0 exactly), LoRA on all linear layers.
  CSO's public command: LlamaFactory --stage dpo, lr 5e-6, per-device batch 4, 3 epochs (other values = LlamaFactory
  LoRA-DPO defaults: lora_rank 8, pref_beta 0.1, sigmoid loss, cosine schedule, warmup 0.1, bf16).
Baseline A, balanced corrective SFT: trl.SFTTrainer on prompt-completion records (loss on the completion only),
  identical LoRA, same number of examples, epochs and batch size as Baseline B.
Both: same init (pi0), resumable (checkpoints every `save_steps`; reruns resume from the latest checkpoint).
"""
from __future__ import annotations

import glob
import json
import os

STUDY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))

CSO_DEFAULTS = {"learning_rate": 5e-6, "per_device_train_batch_size": 4, "num_train_epochs": 3, "beta": 0.1,
                "loss_type": "sigmoid", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                "lr_scheduler_type": "cosine", "warmup_ratio": 0.1}


def lora_config(r=8, alpha=16, dropout=0.0):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")


def latest_checkpoint(out):
    c = sorted(glob.glob(os.path.join(out, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    return c[-1] if c else None


def load_model(path, device_map="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device_map)
    return model, tok


def train_dpo(model, tok, records, out, lr, seed, epochs=3, batch=4, beta=0.1, max_length=12288, save_steps=50, r=8, precompute_ref=False):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    cfg = DPOConfig(output_dir=out, learning_rate=lr, num_train_epochs=epochs, per_device_train_batch_size=batch,
                    gradient_accumulation_steps=1, beta=beta, loss_type="sigmoid", lr_scheduler_type="cosine",
                    warmup_ratio=0.1, bf16=True, gradient_checkpointing=True, max_length=max_length,
                    max_prompt_length=None, max_completion_length=512, use_logits_to_keep=True,
                    logging_steps=5, save_strategy="steps", save_steps=save_steps, save_total_limit=3,
                    report_to=[], seed=seed, data_seed=seed, dataset_num_proc=4, remove_unused_columns=False,
                    precompute_ref_log_probs=precompute_ref)
    ds = Dataset.from_list([{k: x[k] for k in ("prompt", "chosen", "rejected")} for x in records])
    tr = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora_config(r, 2 * r))
    tr.train(resume_from_checkpoint=latest_checkpoint(out))
    tr.save_model(os.path.join(out, "adapter"))
    json.dump(tr.state.log_history, open(os.path.join(out, "log_history.json"), "w"), indent=1)
    return tr


def train_sft(model, tok, records, out, lr, seed, epochs=3, batch=4, max_length=12288, save_steps=50, r=8):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(output_dir=out, learning_rate=lr, num_train_epochs=epochs, per_device_train_batch_size=batch,
                    gradient_accumulation_steps=1, lr_scheduler_type="cosine", warmup_ratio=0.1, bf16=True,
                    gradient_checkpointing=True, max_length=max_length, completion_only_loss=True,
                    logging_steps=5, save_strategy="steps", save_steps=save_steps, save_total_limit=3,
                    report_to=[], seed=seed, data_seed=seed, dataset_num_proc=4)
    ds = Dataset.from_list([{"prompt": x["prompt"], "completion": x["completion"]} for x in records])
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora_config(r, 2 * r))
    tr.train(resume_from_checkpoint=latest_checkpoint(out))
    tr.save_model(os.path.join(out, "adapter"))
    json.dump(tr.state.log_history, open(os.path.join(out, "log_history.json"), "w"), indent=1)
    return tr
