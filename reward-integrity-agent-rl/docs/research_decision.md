# Research decision: "When Rewards Lie" (2026-11-18)

**Decision: STOP.** This is a stop on the proposed method and on the multi-step-audit contribution. No RL was run; none was authorized.

## Gates
| Gate | Result | Evidence |
|---|---|---|
| Literature novelty | **FAIL for a method** | Budgeted audits in GRPO (Audit Where Hacks Pay), audit-based selective control (2609.35677) and negative-only learning (NOPO) cover response-level verifier errors. The open multi-step angle is cheaply closable (next row) |
| Benchmark validity | **PASS (CPU)** | 2 workflow families, 5 legitimate solution styles pass all verifiers (0 false negatives), diagnostics behave as designed, the strong verifier matches independent recomputation; 24 tests |
| Realistic mismatch | **PASS in synthetic workers; real LM untested** | Non-adversarial stochastic worker (10% slips): 64–68% of W1-accepted and 55–61% of W2-accepted outputs are wrong. Mechanisms: wrote-unread-record, omitted-records (multi-step) and miscomputed (single-step) |
| Cheap deterministic correction | **Triggers STOP for the multi-step angle** | A trace check (all listed ids read and written; no source values used) cuts the FP share by about half and leaves **only** single-step miscomputation (100% of residual FPs, both families, all slip rates) |
| Frozen LM pilot (Stage A) | Not run | GPU use was not authorized for this project |
| RL (Stage B) | Not run | Gates failed; no authorization |

## Answers
- **A. Novelty:**
  - The gap is budgeted auditing for errors created by combinations of tool actions, plus on-policy drift of FP types.
  - Prior work solves single-response exploitation under budgets (LB, selective control, NOPO).
  - The combination gap is real in the literature, but in an executable tool environment it is cheaply removed by a deterministic process check over the tool trace.
  - What would remain is ordinary response-level error, which the prior methods handle.
- **B. Benchmark:**
  - Families: invoice and inventory, 3–6 records each.
  - Weak verifiers: W1 checks report/ledger self-consistency; W2 adds a one-record spot check.
  - Strong verifier: all values recomputed from source, plus an exact report.
  - Validated on 2 × 200 tasks.
- **C. Feasibility:** the disagreement is measured only with scripted and stochastic workers. No LM rollouts.
- **D. RL experiment** (planned, not authorized): see experimental_protocol.md. About 10 H100-h (~$25–30) for 2 seeds × {strong, W1, W2} at 1.5B.
- **E. Recommendation: STOP.** Do not develop a multi-step audit-allocation method.
  - If the user wants to keep the question alive, run the single invalidating experiment below (≈ 0.3 GPU-h) before any RL.

## Smallest experiment that can invalidate the central hypothesis
- **Setup:** frozen Qwen2.5-1.5B-Instruct, 2 families × 200 tasks × 8 samples, on this benchmark.
- **Measure:** the share of weak-accepted-but-wrong trajectories that **survive the deterministic trace check**.
- **Stop condition:** if it is ≈ all single-step miscomputation (as with synthetic workers), the multi-step hypothesis is invalid and STOP is final.
- **Reopen condition:** only if a substantial fraction of surviving FPs are composition errors invisible to trace checks (e.g. a value read for one record written to another) would H2/H4 be worth an RL run.
