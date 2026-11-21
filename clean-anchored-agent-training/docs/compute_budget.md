# Compute and storage budget

## Measured on CPU (this session)
- **Token lengths.** Measured with the real Qwen3-4B-Instruct-2507 tokenizer and chat template (`scripts/measure_budget.py --sample 25`; files in `results/budget/`).
  - 0 prefix-stability failures, so the loss masks are valid.
  - The longest sequence is 5,997 tokens; nothing is dropped at max_len 12,288.
  - The mean is about 4.3k tokens per sequence, which is mostly the tool-schema system prompt.
  - Action targets are about 35–100 tokens.

  | arm | primary examples | primary tokens / epoch (extrapolated) | anchors | anchor tokens / pass |
  |---|---|---|---|---|
  | STAGE0 | 748 trajectories | 3.2 M | — | — |
  | EXP1 | 786 actions | 3.5 M | — | — |
  | EXP2 | 709 pairs (1,418 sequences) | 6.1 M | — | — |
  | EXP3 | 709 pairs | 6.1 M | 524 twins | 2.5 M |
  | B7_replay | 709 pairs | 6.1 M | 492 generic (capped to the twin count) | 1.9 M |

- **Data construction cost** (manifest): 8,632 environment tool executions for all train states, branches and twins; 6,727 more for the dev, test and held-out evaluation states. No model calls. About 7 s of CPU.
- **CPU validation** (tests, environment validation, scripted evaluations): under 10 minutes in total.

## GPU estimate (1× H100 80GB). Assumptions are stated; the real number is measured in step 2 of the pilot
**Throughput assumption:** about 1.1 s per 4k-token sequence for forward + backward. This was measured in the previous project: SFT of 744 trajectories took 823 s at peak 23.5 GB, with LoRA r32, bf16, gradient checkpointing and micro-batch 1. Evaluation of 828 episodes took about 48 s of generation plus about 3 min of vLLM and LoRA load; the new suite has about 2,200 episodes, so about 2 min of generation plus 3 min of load.

| step | work | est. H100-h |
|---|---|---|
| setup: pip, 8 GB download, tests, validation, construction | — | 0.5 |
| STAGE0 (748 trajectories, 1 epoch) | 748 sequences | 0.25 |
| eval base + π₀ | 2 × ~2,200 episodes | 0.2 |
| premise gate | 16 rounds × 4 gradient batches | 0.15 |
| policy sampling + verification (K=4) | ~1,100 first actions + ~5k branch rollouts | 0.4 |
| **per seed:** EXP1 (180 × 8) / EXP2 (180 × 16) / EXP3 (180 × 24) / B7_replay (180 × 24) | 1,440 / 2,880 / 4,320 / 4,320 sequences | 0.45 / 0.9 / 1.3 / 1.3 = **3.95** |
| per seed: 4 evals | 4 × ~5 min | 0.35 |
| **Pilot total, 3 seeds** | | **≈ 13.4** |

**Budget control** (recommended and pre-registered as a stage gate):
- Run **seed 0 of all four arms first**, which is about 5.8 H100-hours including setup.
- Continue to seeds 1–2 (+8.6) only if seed 0 shows EXP3 − max(other arms) > 0 on held-out genuine repair, or the premise gate passes.
- **Hard ceiling: 15 H100-hours.** At a typical $2–3 per H100-hour that is about $30–45; the actual pod price was not checked here.
- If step-2 throughput is slower than 1.5 s per sequence, cut the pilot to 120 steps. That keeps the arms matched (about −33%).

## Storage (pod volume, not C:)
| item | size |
|---|---|
| base model | 8 GB |
| merged π₀ | 8 GB |
| 12 LoRA adapters (r32, about 130 MB each) | 1.6 GB |
| evaluation traces with raw completions (about 10–15 MB per checkpoint × 14) | 0.2 GB |
| pip and vLLM caches | about 15 GB |

- **Volume:** provision **≥ 60 GB**; 100 GB is comfortable.
- **Before terminating the pod,** back up everything except the two 8 GB models (`results/`, `logs/`, `data/`, adapters; about 2 GB) and verify the sha256 of the backup.

## Local machine
- **Disk:** C: has about 100 MB free and no other drive. Models, checkpoints and datasets must not land on C:.
- **This session's footprint:**
  - repo: 6 MB data + results;
  - scratch: the tokenizer (16 MB) and the bfcl, peft and accelerate libraries.
- The scratch directory may be cleared on reboot. Reinstall with:
  - `pip install --no-deps --target <dir> bfcl-eval==2026.3.23 peft==0.17.1 accelerate==1.10.1`
  - then set `PYTHONPATH=<dir>`.
