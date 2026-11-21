# Method study 7: CAW-v2 (check-all-writes + persistent-failure continuations). Pre-registered 2026-11-21, before training
## Motivation (method_caw.md)
- CAW transferred to both environments:
  - τ² implicit check rate +0.628 [0.397, 0.859] over SYN64, including 28/44 cancellations;
  - BFCL held-out +0.163 [0.081, 0.236].
- But it **regressed the original suites** (test 0.76–0.84 vs B 0.87), concentrated in S2 (persistent outage, where the backup endpoint is correct) and S3 (impossible, where escalation is correct).
- **Diagnosis:** the data never showed what to do when the re-issued call fails again, so the agent loops between checking and retrying.

## Change (the only one)
88 execution-verified persistent-failure continuations on BFCL fold-T train tasks (`scripts/build_caw2_data.py`, `data/caw2_v1`):

| kind | sequence | chosen action | records |
|---|---|---|---|
| `persist_backup` | f → timeout, check → not applied, f → timeout | **f's backup endpoint** | 44 |
| `persist_escalate` | f → timeout, check → error, check → error | **escalate** | 44 |

- **Verification:** each chosen continuation succeeds under the grader, and blind retry fails.
- **Size:** they replace 88 cycled BFCL records, so the total is unchanged (1,349 + 84 anchors = 1,433). The synthetic part is identical to CAW.
- **Training:** SFT from π_B, identical hyperparameters, **seeds 0 and 1**.

## Rules (fixed now)
**CAW-v2 claimed** only if all hold, pooled over its 2 seeds:
- (i) τ² implicit check rate − SYN64 > 0, CI excluding 0;
- (ii) BFCL held-out Travel implicit success not lower than SYN64 by more than 0.05;
- (iii) S1|S0 ≥ 0.95;
- (iv) **original test and held-out overall each within 0.02 of B_s0 (0.872 / 0.732)**, pooled mean over seeds.

**Diagnostic, reported regardless:**
- test and held-out S2 and S3 vs B and CAW;
- CAW-v2 − CAW on every primary metric;
- over-checking on τ² control writes.

**Budget:** 2 pods in parallel, about 2 h, about $17.

## Result, 2 seeds (2026-11-21; `pod_backups/caw2_2seeds_slim_20261121.tgz`, sha256 59812320…)
| condition | CAW-v2 (s0 / s1) | reference | verdict |
|---|---|---|---|
| (i) τ² implicit check rate | **0.846** (0.692 / 1.000) | SYN64 0.026; CAW 0.654 | **+0.821 [0.667, 0.962] vs SYN64** ✅ |
| (ii) BFCL held-out Travel implicit | 0.750 (0.743 / 0.757) | SYN64 0.667 | +0.083 [0.002, 0.155] ✅ (not lower) |
| (iii) S1\|S0 | 0.95 / 0.99 | ≥ 0.95 | ✅ |
| (iv) original test, pooled mean | **0.878** (0.856 / 0.900) | B 0.872 | +0.006 ✅ |
| (iv) original held-out, pooled mean | **0.727** (0.728 / 0.726) | B 0.732 | −0.005 ✅ |

**Pre-registered claim rule: MET. CAW-v2 is claimed** (2 seeds).

**Diagnostics:**
- **The regression is fixed where it occurred:**

  | suite | CAW-v2 | CAW | B |
  |---|---|---|---|
  | test S3 | **1.000 / 1.000** | 0.775 / 0.767 | 0.930 |
  | test S2 | 0.806 / 0.868 | 0.791 / 0.543 | 0.783 |

- **τ² by function:**
  - `cancel_reservation` **38/44** (CAW 28/44; SYN64 0/66);
  - `book_reservation` 6/8.
- **CAW-v2 − CAW:**
  - τ² +0.192 [0.038, 0.385];
  - BFCL held-out implicit −0.080 [−0.150, −0.007] (CAW 0.829 → 0.750).

  This trade-off is reported.
- **Over-checking on τ² control writes:** check rate 0.694, but mean calls 4.33 vs 5.64 for SYN64. The checks replace failed retries rather than adding cost.
