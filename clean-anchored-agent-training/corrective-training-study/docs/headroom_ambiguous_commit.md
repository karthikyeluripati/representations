# Direction: ambiguous-commit faults (pre-registered 2026-11-20, before any model is evaluated on them)

## Why
- **Headroom over verified DPO (B) on the current suites is about +0.03** held-out overall (results.md, "Headroom over B"). 86% of B's stable failures are competence-limited.
- **B's one recovery-specific weakness is over-persistence:** it re-issues a failed call instead of switching endpoint or escalating.
- **None of the current faults makes retrying harmful.** Every outage string signals "not applied", so retry, backup and escalate cover them all.

## The fault
The task's fault target f (its first state-changing call) returns, on its first call only:

> Error: request timed out after the server received it; the operation may or may not have been applied.

| variant | what happened | correct behaviour |
|---|---|---|
| **ACa** (applied) | f executed | check state, do **not** re-issue f, finish the turn |
| **ACl** (lost) | f did not execute | check state, re-issue f, finish the turn |

The observation is identical in both, so only verify-then-decide passes both. Grading is BFCL's state equality, unchanged: a duplicate booking, order, message or payment is a wrong end state.

## Feasibility (CPU; `scripts/feasibility_ambiguous_commit.py`)
- **A task is diagnostic** if both hold:
  - executing f twice breaks the end state;
  - a read-only function of f's class, with arguments available in context, shows the difference.
- **116 of 398 usable tasks qualify:** train 71, dev 20, test 17, held-out 8. Fault functions include book_flight 34, place_order 29, send_message 20, purchase_insurance 10, plus fund, withdraw and ticket calls.
- **The held-out family (VehicleControlAPI) contributes none.** Its actions are idempotent setters.
- **Scripted validation (all 116 tasks):**

  | policy | ACa | ACl | S0 |
  |---|---|---|---|
  | verify-then-decide | 116/116 | 116/116 | 116/116 |
  | blind retry | 0/116 | 116/116 | 116/116 |
  | never retry | 116/116 | 0/116 | 116/116 |
  | escalate | 0 | 0 | 116/116 |

- The parent test suite passes (27/27) with the fault added (`src/environment/bfcl_env.py`, scenarios ACa/ACl).

## Headroom measurement (GPU, about 1 h; `scripts/eval_ambiguous_commit.py`)
- **Policies:** base Qwen3-4B-Instruct-2507, π₀ (volume `9ghq6ztmfl`), B_s0–s2, A_s0–s2. All frozen; nothing is trained or selected.
- **Episodes:** the 116 diagnostic tasks × {S0, ACa, ACl}, 348 per policy, greedy.
- **Primary metric:** `AC_mean_given_S0`, the mean of ACa and ACl success over tasks the policy solves fault-free. This separates recovery from competence. Reference points: blind retry = 0.50, never retry = 0.50, verify-then-decide = 1.00.

**Gate (fixed now), on B's mean over 3 seeds:**
- **≤ 0.75: headroom confirmed.** At least 0.25 is recoverable, about 8× the current-suite headroom. Proceed to a method design, pre-registered separately.
- **≥ 0.85:** B already handles ambiguity. Drop the direction.
- **Between:** marginal. Report and decide with the user.
- **Also reported:** the action taken right after the ambiguous error (verify / retry / escalate / stop / other). Whether the base model verifies more than π₀/B tells us if corrective training *removed* a skill.

**Known limitation for a later method study:** only 8 held-out-family tasks qualify. A method evaluation will need a new pre-registered split, or more fault positions per turn, defined before training.

## Result (H100, 2026-11-20; `pod_backups/ac_headroom_20261120.tgz`, sha256 4e7809a4…)
116 diagnostic tasks × {S0, ACa, ACl}; greedy; frozen policies.

| policy | S0 | ACa | ACl | **AC mean given S0** | action after the ambiguous error |
|---|---|---|---|---|---|
| base | 0.819 | 0.026 | 0.784 | **0.489** | retry 101, verify 2, stopped 3, no error seen 10 |
| π₀ | 0.845 | 0.000 | 0.845 | **0.500** | retry 113 |
| A s0/s1/s2 | 0.888/0.897/0.914 | 0.000/0.009/0.009 | 0.888/0.897/0.897 | **0.500/0.505/0.495** | retry 115 |
| B s0/s1/s2 | 0.888/0.897/0.888 | 0.017/0.017/0.017 | 0.888/0.897/0.888 | **0.510/0.510/0.510** | retry 115 |

- **Gate:** B mean = **0.510 ≤ 0.75**, so **headroom confirmed.** About 0.49 is recoverable, against about 0.03 on the current suites.
- **Every policy re-issues the call blindly**, even though the error message says it may already have been applied. The base model verifies in 2/116 cases, so the skill is absent before any corrective training, not removed by it.
- **Next gate, not yet run:** a **prompting baseline** (an instruction to check state after an ambiguous failure). If prompting alone closes most of the gap, this is not a training-method problem.
