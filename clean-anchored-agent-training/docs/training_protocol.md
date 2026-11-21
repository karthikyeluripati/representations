# Training protocol: three-arm pilot (pre-registered 2026-11-19, before any training)

## 1. Model and initialization
- **Base:** Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f17c01a7cc42f958dc650907174af0554.
- **π₀ (stage 0)** = base + LoRA (r 32, α 64, all attention and MLP projections) trained 1 epoch at lr 1e-4 on 187 fault-free + 561 injected-outage (S1/S2/S3) reference trajectories from train tasks, then merged. This reproduces the conditions of the previous project's over-generalising recovery SFT (744 trajectories), so the pilot starts from a policy that exhibits the problem.
- **Requirement:** π₀'s held-out false-outage rate is **measured** (step 3 of `scripts/run_pilot_h100.sh`) before any arm runs. If π₀ does not over-generalise (false-outage rate below 0.15), the pilot has nothing to fix and stops there.

## 2. Arms (all from π₀, a fresh LoRA r 32, identical optimizer settings)
| arm | objective | data | anchors |
|---|---|---|---|
| EXP1 | action-level NLL | a+ at every corrective state + the twin-anchor actions (the same states and targets as EXP3) | — |
| EXP2 | DPO (β 0.1) at verified divergent steps (= CSO, one offline round) | verified pairs | — |
| EXP3 (CACRT) | DPO + λ·NLL on twins (λ 1) | verified pairs | counterfactual twins (T-out, T-clean) |
| B7_replay | DPO + λ·NLL on generic anchors | verified pairs | generic clean ground-truth steps, pool capped to the twin pool size |

**Shared settings:**
- **Optimization:** lr 2e-5 with cosine schedule and 10% warm-up; 180 optimizer steps; 8 primary examples per step, plus 8 anchors per step for anchored arms; gradient-norm clip 1.0; max length 12,288; seeds 0, 1, 2.
- **Matching:** EXP3 and B7_replay process exactly the same number of pairs and anchors.
- **Budget differences:** EXP1 and EXP2 see fewer tokens per step. Every run records this in `data_summary.json` and `tokens_seen`.
- **Pairs:** policy-verified (π₀-sampled negatives, π₀ continuations, K=4) if at least 100 corrective states retain a pair; otherwise the CPU surrogate-verified pairs, recorded in `logs/pair_source.txt`.
- **Anchors:** only twins that π₀ already answers correctly greedily (`--policy-correct-anchors`).

## 3. Model selection
- **No selection.** The final checkpoint of every run is evaluated, once, on test and held-out.
- **Dev:** dev corrective states and dev twins are used only for training curves (decision accuracy every 30 steps).

## 4. Evaluation
- **Episodes** (from `src/evaluation/suites.py`):

  | split | total | S0–S3 and shift suites | RP + RA |
  |---|---|---|---|
  | held-out | 1,582 | 114 tasks × 13 | 100 |
  | test | 619 | — | 60 |

- **Decoding:** greedy, at most 20 steps and 20 tool calls per episode.
- **Metrics:** fault-free success; genuine-repair success (RP, RA, and both); injected recovery; conditional recovery (on tasks whose S0 episode succeeded); correct escalation; premature abstention; false-outage rate; observation-shift (S1_obs, S2_obs) and action-shift (S2_act) success; S0 retention relative to π₀; model calls; tokens.
- **Uncertainty:** paired base-task cluster bootstrap, pooled over the 3 seeds, with per-seed values also reported.

## 5. Decision rules (fixed now)
- **GO to the full study** requires all four:
  1. **Held-out genuine repair:** EXP3 exceeds max(EXP1, EXP2, B7_replay) by ≥ 0.05, with the paired 95% CI excluding 0.
  2. **Fault-free guard:** EXP3's fault-free success on test and on held-out is ≥ π₀'s − 0.02.
  3. **Both error types:** the gain is present on both RP and RA, with point estimates > 0.
  4. **Twin-specific gain:** EXP3 > B7_replay on held-out genuine repair *or* false-outage rate, with the CI excluding 0.
- **STOP (method)** if either of these holds:
  - EXP3 is not distinguishable from EXP1, i.e. balanced SFT on the same states performs the same;
  - EXP3 is not distinguishable from B7_replay, i.e. any anchor works.
  In that case report the strongest simple baseline and its numbers.
- **Premise gate** (before the arms run): if mean cos(∇DPO, ∇twin) is not below mean cos(∇DPO, ∇generic) on π₀, the twin component has no mechanistic basis. The arms still run, because the result is informative, but the expectation becomes STOP.
- **Power caveat:** held-out repair has 100 states in 44 base tasks. A 5-point difference is **not** reliably detectable. Expect to detect only differences of roughly 15 points or more. A null result at 5 points is inconclusive, not evidence of equivalence.
