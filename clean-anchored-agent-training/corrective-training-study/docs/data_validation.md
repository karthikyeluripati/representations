# Data validation (CPU, 2026-11-19)

## Archive
- **File:** `representations/pod_backups/pod_backup_20261119.tgz`.
- **Integrity:** sha256 a460d0fa… verified. Read in memory only; nothing was extracted except one file, listed below.
- **Archived manifests:** every file in `cacrt_pairs_v1` and `cacrt_pairs_v1_policy` hashes correctly.
- **Task file:** `tasks_v1.json` sha256 1f912534… equals the local copy (the UTF-8 build).
- **Summaries:** the archived base and π₀ summaries recompute **exactly** from the per-episode traces with the current metric code.
- **Recovered:** `data/policy_pairs_v1.jsonl`, sha256 762953f8…, byte-identical to the archived file.

## Verified pairs
- **Counts:** 567 pairs over 252 distinct states, all on TRAIN tasks.
- **Negatives:**

  | negative | pairs |
  |---|---|
  | escalate | 252 |
  | alternative endpoint | 144 |
  | retry | 123 |
  | π₀-sampled | 48 |

- **Rollouts:** each branch has K=4 π₀ continuations from bit-identical replayed states.
  - Chosen branches succeeded in 2,266 of 2,268 rollouts; rejected branches in 0 of 2,268.
  - Every pair passes the pre-registered `verify_stochastic` rule and also the official CSO rule (the chosen branch reaches success).
- **Caveat.** 252 of the 567 pairs have "escalate" as the negative: an easy contrast that ends the episode. The genuinely discriminative negatives are the 123 retry, 144 alternative-endpoint and 48 sampled pairs.

## Tokenization (real Qwen3-4B-Instruct-2507 tokenizer and chat template)
- **Check:** all 567 × 2 DPO completions and all 567 SFT decisions satisfy tokens(prompt) + tokens(completion) + [eos] = the template's joint tokens. There are 0 failures (`results/dataset_dry_run.json`).
- **Lengths:** the longest prompt is 5,941 tokens; nothing is truncated at max_length 12,288.

## TRL objective checks (`tests/test_trl_objectives.py`; tiny random Qwen3 with the real tokenizer and real records)
- TRL `concatenated_forward` chosen/rejected log-probs equal an independent per-token sum to within 1e-3.
- At initialization (LoRA B = 0, so policy = reference π₀) the DPO loss is log 2.
- Six DPO steps increase the reward margin.
- The SFT completion mask covers exactly the completion tokens plus `<|im_end|>`, and the prompt tokens are unchanged.

## Benchmark
- **Parent checks:** environment validation and all 27 unit tests passed on the pod and locally.
- **Reproducibility:** `PYTHONHASHSEED=0` is pinned, so Windows and Linux builds are byte-identical.
- **Overlap:** the held-out family shares no ground-truth sequence with train, and no superseded result is read anywhere.
