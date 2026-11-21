# Compute budget: tool-interface invariance (2026-11-18)

The previous project's budget is in `docs/archive/alignment_exploration_2026-11-18/compute_budget.md`.

## Resources inspected
- **Local:**
  - Windows 11, 12 logical CPUs, 7.2 GB RAM (about 0.9 GB free), torch 2.9.0+cpu, no CUDA.
  - **The C: drive was full (≈ 50 MB free)**, so no model could be loaded or cached locally.
  - Used for literature, code reading, writing scripts, and receiving small results.
- **Pod (user-provided) `ifk83rgizoilvs`:**
  - 1× H100 80GB HBM3, 208 CPUs, 2 TB RAM, 50 GB overlay disk;
  - torch 2.8.0+cu128, transformers 4.57.1.

## What the GPU was used for, and why
| Workload | Hypothesis resolved | Size | Measured time |
|---|---|---|---|
| 7B feature extraction (faithful RepExp features) | Gate C (does the published bonus treat equivalent calls as novel?) | 180 tasks, 7 prompts each, about 9,700 prompt + response forwards; prompts 1.7–2.6k tokens | 38 min |
| 1.5B feature extraction | Gate C robustness across model scale | same | 33 min |
| 7B sampling (16 per prompt, T = 1) + features | Gate D (prevalence and decisions on real samples) | 5,760 samples; 626,816 generated tokens; 29.05M tokens in feature forwards | 72.5 min |
| Analysis | all gates | numpy, sklearn | about 1 min per run (pod CPU) |

- Total pod wall time is about 2.25 h, with some jobs overlapping. That's roughly **$6–7**, assuming
  $2.7–3/h for an H100; the rate was not verified.
- Peak memory: about 54 GB for 7B extraction (batched prompt + response with 2–3k-token catalogs).
- Running sampling at the same time as extraction caused an out-of-memory error. It was fixed with
  `--gen-batch 8 --chunk 2`.
- **Inefficiencies we accepted:**
  - HF `generate` instead of vLLM: about 33 s per task for 32 samples;
  - a per-response `lm_head` loop.
  - vLLM plus prefix caching would cut sampling roughly 5–10×. It wasn't worth the setup for a one-off
    run.

## Cost of the method under study vs the alternative
- **RepExp bonus:** one extra prompt + response forward per rollout through a frozen copy of the policy.
  With tool catalogs in the prompt that's about 5,040 tokens per sample, against about 109 generated tokens.
  In RL it is an additional reward-model pass every step.
- **Canonical novelty:** one schema decode per response (microseconds, CPU). It was as good as the oracle to
  within 0.03–0.08 unique states.

## Next GPU work
**None justified.**
- **Gate E** (learned invariant representation): canonicalization already reaches the oracle, so there's
  nothing to learn on this benchmark.
- **RL comparison:** about 5 arms × 3 seeds of GRPO at 4–7B, tens of H100-hours. Not justified. The faithful
  within-group bonus is nearly constant, so its effect on the RL advantage is about 0.005, and the fix is
  free.
- **The pod was stopped** after the results were copied.
  - The feature `.npz` files (about 334 MB) weren't copied because the local disk is full.
  - Regenerating them takes about 70 GPU-minutes with the logged commands.
