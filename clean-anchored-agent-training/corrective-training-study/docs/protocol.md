# Protocol: two matched baselines (pre-registered 2026-11-19; machine-readable copy in `configs/baselines.json`)

## Starting policy
- **π₀** is rebuilt on the GPU with the archived STAGE0 config: Qwen3-4B-Instruct-2507 @cdbee75, plus LoRA r32 SFT on 187 fault-free and 561 injected-outage train reference trajectories (lr 1e-4, 1 epoch, seed 0), then merged.
- **Reproduction gate** (`scripts/compare_reproduction.py`). Reuse the archived 567 verified pairs only if both conditions hold on test *and* held-out:
  - episode agreement with the archived π₀ evaluation is ≥ 0.90;
  - every headline paired-difference CI contains 0.
- **Otherwise** the pairs are regenerated under the rebuilt π₀ (same K=4 verification) and both datasets are rebuilt. The choice is logged.

## Arms (identical init, LoRA r 8 on all linear layers, 3 epochs, batch 4, cosine schedule with 10% warm-up, seed 0)
| arm | framework | data | lr grid (dev only) |
|---|---|---|---|
| A: balanced corrective SFT | TRL SFTTrainer, completion-only NLL | 567 verified decisions: 189 real-error repairs + 189 injected-outage responses + 189 fault-free steps | 1e-5, 1e-4 |
| B: CSO-style verified DPO | TRL DPOTrainer, sigmoid, β 0.1, reference = π₀ | 567 π₀-verified pairs (252 states) | 5e-6 (CSO default), 5e-5 |

**Matching:**
- **Examples and updates are identical:** 567 examples × 3 epochs → 425 steps per arm.
- **Prompt tokens are matched:** 2.45 M vs 2.49 M per epoch.
- **Supervised tokens differ, by design of the losses:** A trains on 22.7 k completion tokens per epoch; B scores 32.0 k chosen + 25.4 k rejected tokens per epoch.

## Selection and evaluation
- **Selection:** dev only. Choose the highest dev `overall` (macro success over suites); for ties within 0.005, the lower false-outage rate.
- **Final evaluation:** test and held-out are evaluated **once** per selected frozen adapter, alongside the rebuilt π₀.
- **Fixed for all arms:** greedy decoding, 20 steps and 20 calls, and the identical graders and suites.

  | suite | episodes |
  |---|---|
  | S0 | 1 per task |
  | S1/S2/S3 | × 3 seeds each |
  | S1_obs, S2_obs | reserved EVAL outage strings |
  | S2_act | reserved `_replica` alternative naming |
  | RP | 88 real precondition errors |
  | RA | 72 real argument errors |

  Test has 619 episodes and held-out 1,582.
- **Primary outcome:** held-out overall success.
- **Secondary outcomes:**
  - genuine repair (RP and RA separately);
  - false-outage rate;
  - fault-free retention vs π₀;
  - correct escalation;
  - premature abstention;
  - unneeded retries;
  - outage responses to *unseen* real-error types;
  - shift suites;
  - recognition-probe accuracy (real vs outage);
  - tokens and GPU-hours.
- **Statistics:** paired, base-task cluster bootstrap (1,000 resamples) for B vs A, A vs π₀ and B vs π₀. Raw counts are reported for every suite. With **one seed**, results are a pilot; nothing is claimed for publication from them.

## Diagnostics: question 5 (recognition vs choice)
- **Taxonomy:** `analyze_failure_types.py` labels every failed episode with A–H by precedence and records label overlap.
- **Probe:** `src/analysis/probe.py` asks for one letter at each real-error state and its outage twin.
- **Interpretation:**
  - a correct letter with an outage action means **recognises but chooses wrongly**;
  - a wrong letter means **does not recognise**.
- These are descriptive measurements. Causal claims need a training intervention.

## Controls (from the brief)
- Same start model; matched examples and updates; verified graders; same held-out tasks.
- No test or held-out data in training or selection.
- No new thresholds after seeing results.
- No oracle at inference.
- Competence (S0) and recovery are reported separately.

## Replication protocol (pre-registered 2026-11-19, after seed 0 and before any replication result)
- **Why it's needed:** the seed-0 result (B − A held-out overall −0.031 [−0.055, −0.008]) is one training seed from one π₀. The exact π₀ was lost when the pod stopped, so seeds 1–2 cannot reuse it.
- **Design.** One π₀ is rebuilt **once** with the archived STAGE0 config, stored on a RunPod network volume, and checksummed. Pairs are re-verified under it (K = 4, same rule) and both datasets rebuilt. A and B are then retrained for **seeds 0, 1, 2** with the seed-0 hyperparameters, which are not re-tuned:
  - A: balanced SFT, lr 1e-4;
  - B: verified DPO, lr 5e-6, reference log-probs precomputed (same objective; unit-tested).
- **Evaluation:** every final adapter is evaluated once on test and held-out. There is no model selection.
- **Primary outcome:** held-out overall success, seed-pooled paired base-task cluster bootstrap (`scripts/pool_seeds.py`), for B − A, A − π₀ and B − π₀.
- **Decision rules (fixed now):**
  - A > B confirmed: pooled B − A CI < 0 and at least 2 of 3 seeds with B − A < 0.
  - Otherwise A vs B is inconclusive.
  - A residual false-outage gap justifies **one** targeted intervention only if, in A, false-outage failures remain the largest held-out failure category in at least 2 of 3 seeds.
- **Skill-CDPO-style weighting is not a separate arm.** On these pairs it is degenerate: 512 of 515 pairs have identical criticality (Δ = 1.0) and 99.4% of the weights lie in [0.95, 1.05], so it equals B. Reported as a finding, not run.
