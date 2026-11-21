# Source pins

| source | pin | licence | used for |
|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | cdbee75f17c01a7cc42f958dc650907174af0554 | Apache-2.0 | base model |
| CSO (kiaia/CSO) | 55e10b4b758f847dea5406823f5f08b5a7e7628a | README says Apache-2.0; no LICENSE file | method audit only; no code copied |
| bfcl-eval | 2026.3.23 (gorilla @6ea5797) | Apache-2.0 | environment and grader |
| TRL | 0.24.0 | Apache-2.0 | DPOTrainer, SFTTrainer |
| transformers / peft / accelerate / vLLM | 4.57.3 / 0.17.1 / 1.10.1 / 0.11.0 (GPU); CPU tests ran on transformers 4.57.1 | Apache-2.0 | training and evaluation |
| τ²-bench (sierra-research) | c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4 | MIT | candidate independent evaluation (not yet run) |
| AppWorld (StonyBrookNLP) | 42b5bcf3cd334fee33f0c37c02070a9f5807add5 | Apache-2.0 | candidate independent evaluation (not yet run) |

**Original vs reproduced outputs:**
- Original archived outputs stay inside `pod_backups/pod_backup_20261119.tgz` and are only read.
- Reproduced outputs go to `corrective-training-study/results/`, under names ending in `_rebuilt` or `_final`.
