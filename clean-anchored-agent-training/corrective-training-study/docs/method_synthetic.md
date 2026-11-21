# Method study 5: synthetic-API diversity for idempotency-aware recovery (SYN). Pre-registered 2026-11-21, before training
**Motivation** (method_iar.md, method_iar_annotations.md):
- IAR's implicit-timeout rule reaches about 96% on trained API classes but **0/76 on a held-out class**, and MCP idempotency annotations do not change that (2/76).
- The one lever that has improved cross-class transfer is **training-class diversity**: post-check reading went from chance with 2 classes to 84% with 4.
- **Hypothesis:** with enough classes (32 synthetic plus 4 real), the model learns the abstract rule (a timeout on a *create-type, non-idempotent* call means check first; timeouts on set/remove/read calls mean retry) instead of function-specific associations.

## Synthetic APIs (`src/verification/synthetic_apis.py`, `scripts/build_synth_data.py`)
- **32 domains** (clinic, harbor, pharmacy, …). Each has one record type and four functions:

  | function | type | example |
  |---|---|---|
  | create | **non-idempotent** | `register_patient`, `fileBerth` |
  | set | idempotent | `setBerthPier` |
  | remove | idempotent | `removeBerth` |
  | list | read-only | `listBerths` |

- **Surface varies per class:** 12 create verbs, snake/camel naming, id-key name, JSON or text output.
- **No synthetic function name equals any BFCL function** (tested).
- **Decisions are execution-verified on a small state machine.** Post-check observations are decisive by construction (the created record's value is listed iff applied).
- **IAR convention** (832 records):

  | records | situation | decision |
  |---|---|---|
  | 128 | create + timeout-type TRAIN error | check |
  | 256 | after the check | finish if applied, re-issue if lost |
  | 64 | create + 503 rejection | retry |
  | 384 | set/remove + any TRAIN error | retry |

- **Old-convention control** (832, cycled): identical, except create + timeout gets a direct retry.

## Arms (fold T; SFT from π_B; LoRA r8; lr 1e-4; 3 epochs; batch 4; seed 0; no annotations)
| arm | data |
|---|---|
| **SYNIAR** (method) | BFCL IAR data (517, method_iar.md) + synthetic IAR (832) + 84 B anchors = 1,433 |
| **SYNOLD** (matched control) | the same BFCL IAR data (517) + synthetic old-convention (832) + 84 B anchors = 1,433 |
| reference | IART_s0 (BFCL IAR only; implicit held-out Travel 0.486, 0/76 checks) |

The two arms differ **only** in how the synthetic create-type timeouts are answered.

## Rules (fixed now)
- **Primary:** implicit-timeout success on held-out TravelAPI (the same 35 π_B-solvable decisive tasks), SYNIAR − SYNOLD.
- **Gate for seeds 1–2.** All must hold:
  - ≥ **+0.15**;
  - para not lower than SYNOLD by more than 0.05;
  - S1|S0 ≥ 0.95.
- **Secondary:**
  - checks after the implicit timeout on held-out Travel (0/76 so far);
  - in-distribution implicit;
  - extra calls on S0/S1;
  - original test + held-out vs B_s0 (0.872 / 0.732).

**Budget:** 2 SFT runs of about 1,075 steps (synthetic prompts are short) + evaluations. About 2.5–3 H100-h, about $12.

## Seed-0 result (2026-11-21; `pod_backups/synth_s0_slim_20261121.tgz`, sha256 70656b05…)
| held-out TravelAPI (35 tasks) | SYNIAR (method) | SYNOLD (control) | π_B | SYNIAR − SYNOLD |
|---|---|---|---|---|
| **implicit (primary)** | **0.700** | 0.514 | 0.514 | **+0.184 [0.071, 0.286]** |
| para | 0.829 | 0.714 | 0.514 | +0.114 [0.014, 0.214] |
| explicit (train wording) | 0.786 | 0.700 | 0.514 | +0.084 [−0.029, 0.171] |
| in-distribution implicit | 0.917 | 0.917 | 0.500 | 0.000 |

**Gate: passed** (all three conditions):
- (i) +0.184 ≥ +0.15;
- (ii) para is higher, not lower;
- (iii) S1|S0 0.99.

**Checks after the implicit timeout on held-out Travel:**
- **SYNIAR 46/76** (correct after the check 36/46);
- SYNOLD 0/76;
- every earlier policy 0/76.

**This is the first intervention that makes implicit-risk recognition transfer to an unseen API class.**

**Costs and regressions:**
- original held-out 0.736 / test 0.861, against B_s0 0.732 / 0.872 (within 0.02);
- SYNOLD 0.717 / 0.830;
- extra calls on S0 +0.12; S1 mean calls 3.70 vs 3.59.

**Next, as pre-registered:** seeds 1 and 2, for the 3-seed claim rule:
- pooled CI > 0;
- at least 2 of 3 seeds positive;
- original overall within 0.02 of B_s0;
- S1|S0 ≥ 0.95.

## 3-seed result (2026-11-21; `pod_backups/synth_3seeds_slim_20261121.tgz`, sha256 b923b323…)
**Held-out TravelAPI, 35 π_B-solvable decisive tasks.** Per-seed values; seed-pooled paired task bootstrap:

| wording | SYNIAR s0 / s1 / s2 | SYNOLD s0 / s1 / s2 | pooled SYNIAR − SYNOLD |
|---|---|---|---|
| **implicit (primary)** | 0.700 / 0.600 / 0.529 | 0.514 / 0.514 / 0.500 | **+0.099 [0.024, 0.171]** |
| para | 0.829 / 0.814 / 0.786 | 0.714 / 0.543 / 0.657 | **+0.171 [0.095, 0.238]** |
| explicit (train wording) | 0.786 / 0.714 / 0.614 | 0.700 / 0.543 / 0.529 | +0.114 [0.043, 0.176] |

- **Checks after the implicit timeout on held-out Travel:**
  - SYNIAR **46 / 50 / 8** of 76 (correct after the check 36 / 31 / 6);
  - SYNOLD 0 / 0 / 0.
- **Original benchmark** (SYNIAR s0 / s1 / s2), against B_s0 (0.732 / 0.872):
  - held-out 0.736 / 0.717 / 0.717, mean 0.723;
  - test 0.861 / 0.867 / 0.887, mean 0.872.
- **S1|S0:** 0.99 / 1.00 / 1.00.
- **Extra calls on S0:** +0.12 / +0.48 / +0.12.

**Pre-registered 3-seed claim rule: MET.**
- (i) pooled CI > 0: [0.024, 0.171];
- (ii) at least 2 of 3 seeds positive: 3 of 3 (+0.186, +0.086, +0.029);
- (iii) original overall within 0.02 of B_s0 (mean held-out −0.009, test 0.000);
- (iv) S1|S0 ≥ 0.95.

**Honest reading:**
- **The effect is real but smaller and more seed-variable than seed 0 suggested.** The primary falls from +0.184 to a pooled +0.099. Seed 2 barely checks on the held-out class (8/76).
- **The unseen-paraphrase effect is the most robust:** +0.17, every seed at 0.79 or above.
- **The implicit-timeout rule is learned partially and unstably.** It is the first transfer ever observed (0/76 under every other condition), but it is not yet a reliable skill.
- **Natural next steps:**
  - more synthetic classes or seeds, to test whether variance shrinks with diversity;
  - a second environment (τ²-bench airline).

## Diversity-scaling study (pre-registered 2026-11-21, before any 64-class training)
**Question:** at fixed data size, does more API-class diversity make the implicit-risk rule transfer more strongly and more reliably (lower seed variance)?

**Design:**
- **SYN64:** **64 synthetic classes × 1 task per kind** (`data/synth_v2_64`).
- **Matched in size and composition:** exactly the same 832 synthetic records and kind mix as the 32-class set (32 × 2 tasks per kind), all 256 synthetic function names distinct (tested).
- **Unchanged:** the same BFCL IAR data and B anchors; SFT from π_B with identical hyperparameters; seeds 0, 1, 2.
- **Compared with:** the existing 3-seed SYNIAR (32 classes) and SYNOLD (control) runs.
- **Diversity is the only change.**

**Rules (fixed now):**
- **Primary:** held-out TravelAPI implicit success, SYN64 − SYN32, seed-pooled bootstrap.
- **Diversity scaling supported** if:
  - (a) pooled SYN64 − SYN32 > 0 with the CI excluding 0; **or**
  - (b) SYN64 − SYN32 ≥ 0 **and** the minimum per-seed SYN64 check count after the implicit timeout on Travel is at least 30/76 (SYN32's worst seed was 8/76), i.e. reliability improves.
- **Reported:**
  - SYN64 − SYNOLD;
  - para;
  - per-seed checks;
  - the original suites;
  - S1|S0.

**Budget:** 3 seeds in parallel on 3 pods, about 1.8 h each, about $24.

## Diversity-scaling result (2026-11-21; `pod_backups/synth_analysis_scaling_20261121.json`)
| held-out Travel, implicit | s0 | s1 | s2 | pooled |
|---|---|---|---|---|
| SYN64 (64 classes) | 0.729 | 0.771 | 0.500 | **0.667** |
| SYN32 (32 classes) | 0.700 | 0.600 | 0.529 | 0.610 |
| SYNOLD (control) | 0.514 | 0.514 | 0.500 | 0.510 |

- **SYN64 − SYN32 = +0.058 [0.014, 0.110].** Rule (a) is met: **diversity scaling is supported.**
- SYN64 − SYNOLD = +0.156 [0.095, 0.214].
- **Rule (b), reliability, is not met.** Checks after the implicit timeout were 55 / 64 / **8** of 76. Seed 2 fails in both the 32- and 64-class runs, so the failure is seed-specific (initialisation or data order), not data-specific.
- **Para:** SYN64 0.795 vs SYN32 0.810 (−0.014 [−0.048, 0.024]).
- **No regressions:**
  - test 0.876 / 0.882 / 0.888 (B 0.872);
  - held-out 0.737 / 0.723 / 0.713 (B 0.732);
  - S1|S0 1.00;
  - extra S0 calls +0.10 to +0.53.
