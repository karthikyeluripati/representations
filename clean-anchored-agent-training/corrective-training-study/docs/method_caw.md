# Method study 6: Check-All-Writes (CAW). Pre-registered 2026-11-21, before any CAW training
## Motivation (tau2_transfer.md)
- SYN models transfer the implicit-timeout rule to a held-out BFCL class, but **not to τ²-bench airline**: 0/198 checks before re-cancelling, against 10/36 before re-booking.
- The synthetic curriculum taught *create = check, cancel/set/remove = retry*, so the model keys on **verb semantics**.
- **Idempotency is often undocumented:** τ²'s `cancel_reservation` says only "Cancel the whole reservation", yet it refunds on every call. An agent therefore cannot reliably infer which writes are repeat-safe.

## Rule
**After a timeout-type failure of any state-changing call, check the state, then decide.** Applied → continue or finish; lost → re-issue.

| failure | action |
|---|---|
| timeout on create / set / remove / cancel | check, then decide |
| clean rejection ("503") | retry directly |
| timeout on a read | retry directly |

- **Never wrong**, by construction: verify-then-decide succeeds in both twins for idempotent and non-idempotent writes.
- **Cost:** one extra read call after a timeout on a write.

## Data (`scripts/build_caw_data.py`, `data/caw_v1`; execution-verified; unit tests `tests/test_caw_data.py`)
- **64 synthetic classes** (the same as SYN64) × {create, set, remove} × (check state + 2 post-check states) + 503 retries + read retries = **832** (no drops).
- **BFCL part:** method_iar.md decisions **without** the idempotent-retry records, which contradict CAW. That is 352 unique, cycled to 517.
- **Total 1,349 + 84 B anchors = 1,433,** the same size as SYN64. SFT from π_B; LoRA r8; lr 1e-4; 3 epochs; seeds 0–2.

## Comparison and rules (fixed now)
- **Arms:** CAW64 (3 seeds) vs **SYN64** (3 seeds, existing) and SYNOLD.
- **Primary (τ², zero-shot):** check rate right after the error on the 13 diagnostic airline writes, **implicit wording**, pooled over seeds. Compared as CAW64 − SYN64 with an event bootstrap. Also reported by function (`cancel_reservation` vs `book_reservation`).
- **CAW claimed** if all hold:
  - (i) τ² implicit check rate CAW64 − SYN64 > 0, CI excluding 0;
  - (ii) BFCL held-out Travel implicit success not lower than SYN64 by more than 0.05 (pooled);
  - (iii) S1|S0 ≥ 0.95;
  - (iv) original test and held-out within 0.02 of B_s0.
- **Reported:**
  - over-checking cost (extra calls on BFCL S0 and S1, and on τ² control writes);
  - para and explicit on both environments.

**Budget:** 3 pods in parallel. Each trains (about 75 min), evaluates on BFCL, then evaluates on τ² for its seed. About 2 h, about $27.

## Edit log (2026-11-21, before any CAW training or result)
- **Seeds reduced from 3 to 2 (0 and 1)** for budget reasons, decided before any CAW result exists.
- **The claim rule is otherwise unchanged:**
  - (i) the τ² implicit check-rate CI excludes 0, pooled over the 2 CAW seeds and compared with all 3 SYN64 seeds;
  - (ii)–(iv) as above.
- **Reporting:** results will be presented as 2-seed evidence.

## Result, 2 seeds (2026-11-21; `pod_backups/caw_2seeds_slim_20261121.tgz`, sha256 b9748d80…)
**τ² airline, zero-shot** (13 diagnostic writes; check rate right after the error):

| wording | CAW64 (s0 / s1) | SYN64 (3 seeds) | CAW64 − SYN64 |
|---|---|---|---|
| **implicit (primary)** | **0.654** (0.692 / 0.615) | 0.026 | **+0.628 [0.397, 0.859]** |
| para | 0.885 | 0.051 | +0.833 [0.628, 1.000] |
| explicit | 0.846 | 0.051 | +0.795 [0.641, 0.936] |

- **By function (implicit):**

  | function | CAW64 | SYN64 |
  |---|---|---|
  | `cancel_reservation` | **28/44** | 0/66 |
  | `book_reservation` | 6/8 | 2/12 |

  **The verb-keyed failure is fixed.**
- **Over-checking on τ² control writes:** 0.458 check rate; mean calls 5.26 vs 5.64.

**BFCL held-out TravelAPI** (35 tasks):

| wording | CAW64 (s0 / s1) | SYN64 | CAW64 − SYN64 |
|---|---|---|---|
| implicit | **0.829** (0.829 / 0.829) | 0.667 | **+0.163 [0.081, 0.236]** |
| para | 0.879 | 0.795 | +0.083 [0.031, 0.138] |
| explicit | 0.864 | 0.743 | +0.121 [0.052, 0.191] |

- **Both seeds hold.** No seed collapse: SYN32 and SYN64 each had one seed at 8/76.
- **S1|S0:** 1.00 / 0.97.

**Original benchmark (regression):**
- **Overall:**
  - held-out 0.725 / 0.682 (B 0.732);
  - test 0.837 / 0.756 (B 0.872).
- **The loss is concentrated in S2 and S3:**
  - test S2: 0.791 / 0.543 vs 0.783;
  - test S3: 0.775 / 0.767 vs 0.930.

**Pre-registered claim rule:**

| condition | result |
|---|---|
| (i) τ² CI > 0 | ✅ met |
| (ii) BFCL non-inferiority | ✅ met (superior) |
| (iii) S1\|S0 ≥ 0.95 | ✅ met |
| (iv) original overall within 0.02 of B | ❌ **failed** |

**CAW is not claimed.**

**Diagnosis:**
- CAW data teaches "check, then re-issue if not applied", but contains **no persistent-failure continuation**: the case where the re-issue fails again.
- Under persistent outages (S2, where the backup endpoint is correct) and impossible requests (S3, where escalation is correct), the model loops check → retry instead of switching to the backup or escalating.
- **Proposed fix (CAW-v2, to be pre-registered separately):** add execution-verified states in which, after check → re-issue → the same failure, the chosen action is the backup endpoint (S2-type) or escalation (S3-type).
