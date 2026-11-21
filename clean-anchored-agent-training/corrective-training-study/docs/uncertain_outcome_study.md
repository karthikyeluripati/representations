# Uncertain-outcome study: is "retry after an ambiguous failure" a prompting, scale or training problem?
Pre-registered 2026-11-20, after the headroom gate (headroom_ambiguous_commit.md) and before any result below.

## Question
- Every policy tested so far (base 4B, π₀, A×3, B×3) re-issues a non-idempotent call after "may or may not have been applied", in 113–115 of 116 cases. AC success given S0 is 0.49–0.51.
- Before any training method is designed, three questions:
  1. Does **one instruction** fix it?
  2. Does **scale** fix it?
  3. Does it survive **realistic wording**?

## Design (`scripts/eval_uncertain_outcome.py`, `scripts/run_uncertain_outcome_h100.sh`)
- **Tasks:** the 116 diagnostic tasks (`results/feasibility_ambiguous_commit.json`). Greedy, the same agent loop and grader as everything before.

| factor | levels |
|---|---|
| prompt | **std** (the system prompt used in all training and evaluation, which says "If a call fails, you may retry it"); **check** (std plus one sentence: "If a failure leaves it unclear whether an operation already took effect, do not simply repeat it: first check the current state with a read-only tool, and repeat the operation only if it has not taken effect.") |
| suite | S0; S1 (transient outage, retry is correct: the control for over-verification); ACa/ACl × message |
| message | **explicit** ("…may or may not have been applied"); **para** (unseen paraphrase: "connection lost before a response arrived; the request might already have been processed"); **implicit** ("Error: the request timed out.", the realistic case that says nothing about the outcome) |
| policy | B_s0 and π₀ (4B, trained); base Qwen3-4B-Instruct-2507; Qwen3-30B-A3B-Instruct-2507-FP8 @5a5a776 (same instruct family, 7.5× parameters); Qwen3-8B @b968826 (hybrid, thinking disabled) |

- **Volume:** 1,856 episodes per policy.
- **Scripted references, every message:** verify-then-decide = 1.00; blind retry = never retry = 0.50; escalate = 0.

**Metric:** `AC_<msg>_given_S0`, the mean of ACa and ACl success on tasks the same policy solves fault-free under the same prompt. The control is S1 success given S0.

## Decision rules (fixed now)
1. **Prompting suffices** if, under *check*, B_s0 reaches AC ≥ 0.85 on **all three** messages and S1|S0 stays ≥ 0.95. Then a training-method paper on this fault is not justified, and the result is reported as a prompting finding.
2. **Scale suffices** if the 30B model under *std* reaches AC ≥ 0.85 on all three messages.
3. **A training method is justified** only if neither 1 nor 2 holds, i.e. the best of {B+check, 30B+std, 30B+check} still leaves AC ≤ 0.85 on at least one message, or reaches it only by dropping S1|S0 below 0.95. The method study is then pre-registered separately, with a split defined before training.
4. **Reported regardless:**
   - the action distribution after the error;
   - escalations;
   - implicit vs explicit for each policy;
   - whether *check* hurts S0.

## Known limitations
- 116 tasks from 5 API classes; held-out family (VehicleControlAPI) absent.
- The implicit message is genuinely ambiguous. In S1, a "request timed out" string means "not applied". Here the correct robust behaviour (verify) is right in both cases, but blind retry is also right in ACl and wrong only in ACa.

## Results (H100, 2026-11-20; `pod_backups/uncertain_outcome_20261120.tgz`, sha256 a29ff5c6…)
**AC success given S0**, mean of ACa and ACl, 116 tasks per cell:

| policy | prompt | S0 | S1\|S0 (control) | explicit | para | implicit |
|---|---|---|---|---|---|---|
| B_s0 (4B, verified DPO) | std | 0.888 | 1.000 | 0.510 | 0.510 | 0.510 |
| B_s0 | check | 0.888 | 0.981 | 0.505 | 0.505 | 0.505 |
| π₀ | std / check | 0.845 | 1.000 | 0.500 | 0.500 | 0.500 |
| base 4B | std | 0.810 | 0.968 | 0.489 | 0.495 | 0.495 |
| base 4B | check | 0.828 | 0.979 | 0.500 | 0.505 | 0.495 |
| Qwen3-8B (no thinking) | std | 0.759 | 0.943 | 0.472 | 0.489 | 0.477 |
| Qwen3-8B | check | 0.759 | 0.943 | 0.489 | 0.494 | 0.494 |
| Qwen3-30B-A3B-2507 (FP8) | std | 0.888 | 1.000 | 0.500 | 0.500 | 0.500 |
| Qwen3-30B-A3B-2507 | check | 0.897 | 1.000 | **0.534** | **0.649** | 0.500 |

**Action right after the error, ACa** (out of 116):

| policy | std, explicit | check, explicit | check, para | check, implicit |
|---|---|---|---|---|
| B_s0 | retry 115 | retry 115 | retry 114 | retry 115 |
| 30B | retry 113 | verify 12, retry 101 | verify 44, retry 64 | retry 113 |
| base 4B | retry 99, verify 2 | verify 10, retry 93 | verify 24, stopped 54 | retry 106 |

## Decisions (pre-registered rules)
1. **Prompting suffices: not met.** B + check = 0.505 on every message. The instruction has no effect on B, and a small effect on base 4B and 30B.
2. **Scale suffices: not met.** 30B + std = 0.500 on all messages; 8B is no better.
3. **Training method justified: yes.** The best of {B+check, 30B+std, 30B+check} is 0.534 / 0.649 / 0.500, far below 0.85, and S1 is intact.

**Observations:**
- **The unseen paraphrase is easier than the explicit training-style wording** for prompted models. The verb "processed" seems to cue checking more than "applied".
- **The realistic implicit timeout is untouched by prompting or scale** (0.50 everywhere). It is the hardest and most realistic case.
- **Verifying is not sufficient.** 30B + check verifies in 44/116 ACa_para episodes but succeeds in about 40% of ACa_para. Models sometimes re-issue after reading a state that already shows the effect.
