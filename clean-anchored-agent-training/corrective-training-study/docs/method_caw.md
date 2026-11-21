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
