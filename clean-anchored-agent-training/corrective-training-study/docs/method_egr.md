# Method study 2: Evidence-Grounded Recovery (EGR). Pre-registered 2026-11-20, before any EGR training
**Motivation** (method_outcome_twins.md, seed-0 diagnosis):
- Training on uncertain-outcome twins teaches agents to *check*, and checking transfers to unseen API classes (0% to about 58%).
- But the *decision after the check* is at chance on unseen classes, while it is good in-distribution.
- Two likely causes:
  - (i) training covered only 2 API classes;
  - (ii) nothing taught a class-independent rule for reading a check.
- The DPO recipes also regressed ordinary behaviour (test 0.80 / 0.74 vs B 0.87; S1|S0 about 0.90). SFT did not.

## Method
- **Supervised training** on the same execution-verified outcome-twin states, where each target action is preceded by **one sentence of evidence** stating the class-independent attribution rule:
  - **at the error:** "The outcome of `<f>` is uncertain, so I will check the current state before deciding." + check call;
  - **after the check, applied:** "Evidence: the result shows `<value>`, so `<f>` already took effect; I will not repeat it." + continue or finish;
  - **after the check, not applied:** "Evidence: the result does not show `<value>`, so `<f>` did not take effect; I will issue it again." + re-issue.

  `<value>` is the argument of f that the decisiveness criterion found, so every claim is true by construction (unit-tested).
- **Baseline (AAC2): identical** states, actions, anchors, record count and optimiser, **without** the evidence sentences. The arms differ only in the rationale text.
- **Both arms:** start from π_B (persisted, checksummed); LoRA r 8; lr 1e-4; 3 epochs; batch 4.
- **Anchors (both arms):** S1 retry decisions + 84 B-pair chosen actions (seeded).

## Design: 2-fold leave-one-class-out (decisive tasks; train split only for training)
| fold | train classes (train-split tasks) | held-out class (evaluated, all splits) |
|---|---|---|
| **M** | TravelAPI, TradingBot, TwitterAPI, TicketAPI: 34 tasks, 544 verified pairs (340 unique states) | MessageAPI (20 decisive tasks) |
| **T** | MessageAPI, TradingBot, TwitterAPI, TicketAPI: 22 tasks, 352 pairs (220 states) | TravelAPI (38 decisive tasks) |

- **Training wordings:** explicit, train1, train2.
- **Evaluation wordings:** para, implicit (never trained).

## Outcomes and rules (fixed now)
- **Primary:**
  - Per fold: success on ACa/ACl × {para, implicit}, std prompt, over the fold's held-out-class tasks that π_B solves fault-free (a fixed set).
  - **Pooled across both folds** over tasks (`scripts/pool_folds.py`): EGR − AAC2, bootstrap 95% CI.
- **Mechanism (secondary):** the right decision after a check, on held-out classes. This is the diagnosed bottleneck.
- **Seed-0 gate:** seeds 1–2 run only if pooled EGR − AAC2 ≥ **+0.10** at seed 0.
- **EGR claimed (3 seeds)** only if all hold:
  - pooled CI > 0;
  - EGR > AAC2 in at least 2 of 3 seeds;
  - original test and held-out overall each within 0.02 of B_s0 (0.872 / 0.732);
  - S1|S0 ≥ 0.95.
- **Reported regardless:**
  - in-distribution;
  - per-wording results;
  - extra calls on S0;
  - the original suites.

## Budget
- **Seed 0:** 4 SFT runs (about 1.5 h) + evaluations (about 0.7 h) + setup. About 2.5 H100-h, about $11.
- **Each further seed:** about 2.2 h.

## Seed-0 result (2026-11-20; `pod_backups/egr_s0_slim_20261120.tgz`, sha256 30958dbf…)
**Primary** (held-out-class tasks π_B solves fault-free; para + implicit; std prompt):

| | fold M (MessageAPI, 9 tasks) | fold T (TravelAPI, 35 tasks) | pooled (44) |
|---|---|---|---|
| π_B | 0.500 | 0.514 | about 0.51 |
| AAC2 (plain SFT) | 0.611 | **0.686** | **0.670** |
| EGR (evidence SFT) | 0.611 | 0.571 | 0.580 |

- **EGR − AAC2 = −0.091 [−0.148, −0.034].** The gate (≥ +0.10) **failed**, and EGR is significantly *worse*. **EGR is rejected.** Seeds 1–2 not run.
- **Costs of EGR:**
  - lower S0: 0.80 / 0.81 vs 0.86 / 0.90;
  - +1.0 to 1.9 extra calls per fault-free episode (over-checking);
  - original test 0.657 / 0.844 vs AAC2 0.824 / 0.889.

**Check, then decide, on the held-out class:**

| | para: checked | para: right after check | implicit: checked |
|---|---|---|---|
| AAC2 fold T (held-out Travel) | **76 / 76** | **64 / 76** | **0 / 76** |
| EGR fold T | 76 / 76 | 51 / 76 | 0 / 76 |
| AAC2 fold M (held-out Message) | 16 / 40 | 9 / 16 | 4 / 40 |
| EGR fold M | 38 / 40 | 21 / 38 | 0 / 40 |

**Findings (descriptive, single seed):**
1. **Training diversity matters.** Plain SFT on 4 API classes transfers checking *and* reading to an unseen class on the unseen paraphrase: fold T reads 84% correctly. The 2-class OTV run was at chance.
2. **Rationales hurt:** they cause over-checking and worse decisions.
3. **The remaining gap is recognition, not reading.** No policy checks after the realistic *implicit* timeout ("Error: the request timed out."): 0/76 and 0/40. Every training wording states the uncertainty explicitly, and the S1 training convention treats timeouts as "not applied".
4. **Original benchmark:** AAC2 fold T held-out 0.753 / test 0.889, at or above B_s0 (0.732 / 0.872). AAC2 fold M regresses (0.677 / 0.824).
