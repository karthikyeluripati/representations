# Method study: Outcome-Twin Verified Training (OTV). DRAFT pre-registration, 2026-11-20
**Status:** written before any method training. To be frozen, with any edits logged, before the first training run.

## Motivation (from uncertain_outcome_study.md)
- No prompt and no scale up to 30B-A3B fixes blind re-issue after an uncertain failure. The best result is 0.649 on one wording; the realistic implicit timeout gives 0.50 everywhere.
- **Two distinct failures:**
  1. models do not *check* (retry 101–115 of 116);
  2. when they do check, they often *misread* the result and re-issue anyway (30B: 44 checks, about 40% ACa_para success).

## Key idea
**ACa and ACl are twins.** They share the identical history up to and including the ambiguous error, and differ only in the hidden state. So:
- **At the error state** the correct action is the same in both twins: check. Every other action is wrong in at least one twin. This is a preference with a *provably* correct chosen action and *twin-verified* rejected actions.
- **After the check** the twins diverge: the read observation differs, and so does the correct action. The pair teaches *reading the state*, which addresses failure 2. A model cannot fit both twins by ignoring the observation.
- **Anchors keep retry where retry is right:** S1/S2 states (outage strings that unambiguously mean "not applied") and guarded or idempotent calls, where checking is unnecessary. This targets no regression on S0–S3/RP/RA and limits over-verification cost.

**Objective:** step-level DPO (TRL, sigmoid, β 0.1, reference = π₀; identical to B) on three pair types:
- **(a) error-state pairs:** chosen = verify call; rejected = blind retry, and escalate;
- **(b) post-check pairs, per twin:** ACa: chosen = continue, rejected = re-issue. ACl: chosen = re-issue, rejected = continue;
- **(c) anchor pairs:** retry/backup chosen over verify on unambiguous outages, plus the existing B pairs.

The only differences from B are the states and pairs, never the optimiser.

## Arms (all from π₀; identical LoRA r 8, lr 5e-6, 3 epochs, matched number of pairs and steps; seeds 0, 1, 2)
| arm | data on AC train tasks |
|---|---|
| **B** (existing) | none (reference) |
| **B+AC** (CSO-style) | error-state pairs only: chosen = oracle verify-then-decide continuation, rejected = blind retry. No twin post-check pairs, no extra anchors |
| **A+AC** (SFT) | oracle verify-then-decide demonstrations at error and post-check states + the same anchors as OTV, NLL |
| **OTV** (method) | (a) + (b) + (c) |

**Non-trained references (already measured or scripted):**
- prompting (std, check);
- 30B-A3B + check;
- the scripted verify-then-decide oracle (1.00).

## Edit log (before any training)
- **2026-11-20, decisive checks only.** Pair inspection showed that some verifier observations do not reveal the outcome without the earlier state. Example: `fund_account` checked with `get_account_info` shows balance 12203.4 vs 10000.0, which cannot be read without the prior balance.
  - **A task is now *decisive*** if some argument value of f appears in a read-only verifier's output after f and not before.
  - **Training and the primary evaluation use decisive tasks only** (73 of 116), each with its first decisive verifier. All 116 are still reported.
  - **Consequence:** `place_order` (29) leaves the set, because its order history lists only ids and confirming takes two steps.
  - **Resulting counts:**
    - train: 28 tasks (TravelAPI 24, TradingBot 4);
    - unseen-class evaluation: 29 tasks (MessageAPI 20, TwitterAPI 5, TicketAPI 4, all splits);
    - in-distribution evaluation: 16 tasks (Travel/Trading, dev, test and held-out).

## Generalization split (fixed now, before any training)
| axis | train | evaluate |
|---|---|---|
| API class / function | TravelAPI, TradingBot (book_flight, place_order, purchase_insurance, fund_account, …) on **train-split** tasks | **unseen classes:** MessageAPI, TwitterAPI, TicketAPI (send_message, comment, mention, create_ticket, …), all splits (34 tasks); **in-distribution:** Travel/Trading tasks of test/heldout splits |
| wording | explicit + two train-only paraphrases (to be written and frozen with this file) | **unseen:** para, implicit |
| environment | BFCL only | **τ²-bench airline** (book_reservation, cancel_reservation, send_certificate); τ² retail as the guarded-API control (second_environment.md) |

## Outcomes and decision rules
- **Primary:** AC success given S0 on **unseen classes × unseen wordings** (para + implicit pooled). Seed-pooled paired task-cluster bootstrap. OTV vs the best trained baseline (B+AC or A+AC).
- **OTV claimed** only if all hold:
  - (i) primary difference > 0 with a 95% CI excluding 0, and positive in at least 2 of 3 seeds;
  - (ii) no regression on the original held-out overall vs B (CI lower bound > −0.02);
  - (iii) S1|S0 ≥ 0.95.
- **Secondary:**
  - τ²-airline AC success;
  - τ²-retail extra calls (over-verification cost);
  - the ACa post-check misread rate;
  - in-distribution AC success.
- **Ablations, if OTV is claimed:** −(b), −(c), −twins (pairs from ACa only).

## Budget
- **GPU:** 3 trained arms × 3 seeds × about 15–20 min training on a small pair set, plus evaluation. About 4–5 H100-h, about $20. The τ² adapter is a separate step.
- **CPU first:**
  - pair construction with twin verification;
  - wording set frozen;
  - unit tests for twin-pair correctness;
  - the τ² wrapper.

## Edit log 2: starting point, anchors, primary metric, futility gate (2026-11-20, before any training)
- **Start policy for every arm: π_B = π₀ + B_s0 merged** (`scripts/merge_adapter.py`). The DPO reference is π_B (adapter disabled). This replaces "from π₀ + all B pairs":
  - B is the strongest baseline, so the question becomes "does each recipe improve on B?";
  - it avoids about 650-step runs.

  Training seeds vary only the AC-stage training. B itself is near-deterministic across seeds (held-out 0.730–0.732).
- **Arms, matched at 532 records each, 3 epochs, batch 4, 399 steps** (`scripts/render_otv_arms.py`):

  | arm | data |
  |---|---|
  | OTV | (a) 252 + (b) 168 + S1 anchors 28 + 84 B pairs (seeded sample) |
  | BAC | (a) cycled to 420 + the same 28 + 84 |
  | AAC (SFT, lr 1e-4) | chosen actions of the 252 unique (a)/(b) states cycled to 420 + the same 28 + 84 |

  DPO arms: lr 5e-6, β 0.1, reference log-probs precomputed (unit-tested).
- **Primary metric (exact).**
  - Task set: the **29 decisive unseen-class tasks**, restricted to those **π_B solves fault-free under the std prompt**. This is a fixed set, common to all arms.
  - Value: mean success over ACa and ACl, wordings para and implicit, std prompt.
  - Comparison: OTV − best of {BAC, AAC} (best chosen by pooled point estimate), with a seed-pooled paired bootstrap over tasks (2,000 draws).
- **Regression checks:**
  - original test + held-out overall vs B_s0's replication evaluation (lower CI bound > −0.02);
  - S1|S0 ≥ 0.95 on the uncertain-outcome tasks;
  - mean extra calls per S0 episode reported (over-verification cost).
- **Futility gate after seed 0:** seeds 1–2 run **only if** OTV's seed-0 primary metric is above both baselines' seed-0 values. Otherwise stop and report. The claim rule still needs 3 seeds.
- **Budget:** about 3.4 H100-h for seed 0 (setup, merge, render, 2 DPO + 1 SFT runs, evaluations), about 3.2 h per further seed. About $15 for seed 0 and $43 for all three.

## Seed-0 result and futility decision (2026-11-20; `pod_backups/otv_s0_slim_20261120.tgz`, sha256 a4dd26a1…)
- **Primary** (18 decisive unseen-class tasks π_B solves fault-free; para + implicit; std prompt):
  - OTV 0.514, BAC 0.500, AAC 0.528, π_B 0.500.
  - OTV − best (AAC) = −0.015 [−0.111, 0.069].
  - **Futility gate: failed. Seeds 1–2 not run.**
- **In-distribution** (15 Travel/Trading tasks outside the train split): OTV **0.717**, AAC 0.667, BAC 0.567, π_B 0.533.
- **Diagnosis** (first action after the ambiguous error; para + implicit):

  | | checks, unseen classes | right decision after checking, unseen | checks, in-dist | right decision after checking, in-dist |
  |---|---|---|---|---|
  | π_B | 0 / 112 | — | 0 / 64 | — |
  | OTV | 67 / 116 | 28 / 67 (chance-level) | 54 / 60 | 42 / 54 |
  | BAC | 70 / 116 | 28 / 70 | 54 / 60 | 31 / 54 |
  | AAC | 50 / 116 | 20 / 50 | 32 / 64 | 26 / 32 |

  - **Checking transfers** to unseen API classes: from 0% to about 58%.
  - **Reading the check does not.** On unseen classes the post-check decision is at chance. In-distribution, OTV's twin post-check pairs give the best reading (42/54), against 31/54 for BAC.
  - The bottleneck is **outcome attribution on unfamiliar API outputs**: deciding whether a read result reflects one's own pending write.
- **Regressions, all three failing rule (ii) or (iii):**
  - original **test** overall: OTV 0.799, BAC 0.737, AAC 0.867, against B_s0 0.872;
  - **held-out**: 0.722 / 0.703 / 0.727 against 0.732;
  - **S1|S0**: OTV 0.907, BAC 0.897, AAC 0.990.

  The DPO arms damage ordinary behaviour; SFT does not.
- **Limits:** one seed, 18 primary tasks, and training on only 2 API classes (28 tasks). Narrow training diversity is a likely cause of the transfer failure.
