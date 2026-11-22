# CF-SFT pilot, stage 0: prompt selection and baselines (2026-11-22)
**Protocol:** `cf_sft_pilot_protocol.md`, frozen at commit `657f44a`. Nothing in it was changed after inference.

**Evidence:**
- **Archive:** `pod_backups/cf_pilot_stage0_20261122.tgz`, sha256 `c762872ebbb03ac0a5e505b2e6db5dfa7eda705940b4a78544a30279a67210f4`. The checksum is identical on the pod and locally. The archive holds the traces, logs and the exact code tarball (sha256 `68c87f81…`, from `git archive 657f44a`).
- **Report:** `data/cf_pilot_stage0/report.{json,md}`, recomputed locally from the raw traces by `scripts/cf_pilot_stage0_report.py`.

**Run facts:**
- **Hardware and cost:** one H100 SXM, 1,397 s, about **$1.70** (cap $5). The balance is now $4.07.
- **Pre-inference checks:** 33/33 tests passed on the pod; the π_B checksum and the spec and data hashes all matched.
- **Training:** none.

## 1. Prompt selection (T2-dev only: 10 families, 2,160 episodes per run)
Selection rule, pre-registered: the highest dev decision accuracy; ties within 0.01 are broken by fewer excess calls.

| model | P0 (none) | P1 (one-sentence hint) | P2 (four explicit rules) | P3 (P2 + worked example) | **selected** |
|---|---|---|---|---|---|
| π_B: decision accuracy / excess calls | 0.767 / +0.11 | 0.793 / +0.01 | 0.782 / +0.12 | 0.796 / +0.13 | **P1** (within 0.01 of P3, fewer calls) |
| CAW-v2 s0: decision accuracy / excess calls | 0.866 / +0.51 | 0.865 / +0.53 | 0.865 / +0.63 | 0.862 / +0.70 | **P0** |

**On dev, under every prompt and for both models:**
- effect sensitivity, safe escalation and reconciliation are all **0.000**;
- the held-out-combination twin score is 0.500, the same as the scripted always-check and always-retry policies.

**In all 160 unresolvable dev episodes, every run re-issued the non-repeat-safe write.** That includes P2, which states "otherwise call escalate_to_user without repeating the write", and P3, which adds a worked escalation example.

**What the explicit prompts did instead was hurt:**
- **π_B fault-free success:** 0.992 (P0), 0.900 (P2), 0.767 (P3), because it repeated a successful write and then escalated;
- **CAW-v2 fault-free success:** 1.000, 0.903, 0.758;
- **unnecessary checks and excess calls** rose.

The test sets were not touched during selection. P2 and P3 were never run on T2-test or T1.

## 2. Baselines on T2-test (20 families, 4,320 episodes per run)
Pooled over seeds, with a 95% family-cluster CI.

| arm@prompt (runs) | decision accuracy | **effect sensitivity (E2)** | **held-out combos (E1)** | final state, feasible | safe escalation | reconcile | unnecessary checks | excess calls | fault-free |
|---|---|---|---|---|---|---|---|---|---|
| π_B@P0 (1) | 0.666 [0.641, 0.693] | **0.000** | 0.500 | 0.731 [0.703, 0.760] | 0.000 | 0.000 | 0.001 | +0.48 | 0.999 |
| π_B@P1 (1) | 0.688 [0.666, 0.713] | **0.000** | 0.500 | 0.755 [0.731, 0.783] | 0.000 | 0.000 | 0.007 | +0.31 | 0.999 |
| SYN64@P0 (3) | 0.819 [0.814, 0.825] | **0.000** | 0.500 | 0.899 [0.893, 0.906] | 0.000 | 0.000 | 0.317 | +0.37 | 0.999 |
| CAW@P0 (2) | 0.841 [0.836, 0.846] | **0.000** | 0.497 | 0.923 [0.917, 0.928] | 0.000 | 0.000 | 0.641 | +0.72 | 0.998 |
| CAW-v2@P0 (2) | 0.854 [0.847, 0.860] | **0.000** | 0.498 | 0.937 [0.930, 0.943] | 0.000 | 0.000 | 0.641 | +0.56 | 0.999 |
| *scripted contract policy* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.00 | 1.000 |

**Per seed:**

| arm | decision accuracy | final state, feasible | unnecessary checks | E2 (every seed) |
|---|---|---|---|---|
| CAW-v2 (s0 / s1) | 0.849 / 0.859 | 0.932 / 0.942 | 0.643 / 0.638 | 0 |
| SYN64 (s0 / s1 / s2) | 0.835 / 0.820 / 0.803 | 0.916 / 0.900 / 0.881 | 0.492 / 0.309 / 0.148 | 0 |

**π_B@P1 − π_B@P0:** decision accuracy +0.022 [0.013, 0.033]; E2 and E1 +0.000.

## 3. Selected prompts on the frozen T1 test (20 families, 1,920 episodes per run)
| arm@prompt | decision accuracy | effect sensitivity (M1) | history twins (M2) | safe escalation (M3) | unnecessary checks |
|---|---|---|---|---|---|
| π_B@P1 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 |
| CAW-v2@P0 (2) | 0.795 | 0.000 | 0.944 | 0.003 | 0.548 |

**The re-runs reproduce the frozen T1 evaluation:**
- **π_B@P1:** 1,920/1,920 completions identical to the frozen `piB_prompt` run;
- **CAW-v2:** 1,913 and 1,909 of 1,920 identical, with vLLM batch nondeterminism only and no metric change at the reported precision.

## 4. What this means under the pre-registered criteria
- **Stage 0 has no gate** (protocol §5). Its pre-registered role is to fix the prompt-only baseline for S6.
- **Prompting does not close the gap.** On both new held-out sets and on the frozen set, the best dev-selected prompt gives **E2 = 0, E1 = 0.50 and T1 M1 = 0**. That is the floor of an effect-blind policy. The strongest rule prompts also score 0 on dev, while costing fault-free success.
- **The "prompting suffices" outcome (S6) is effectively closed.** S6 needs the prompted baseline within 0.05 of CF on E2, E1 and E3. With the prompted baseline at E2 = 0, S6 could hold only if CF's E2 ≤ 0.05. But CF would then already have failed the stage-1 futility gate (E2 < 0.40). So no stage-1 outcome leads to a "prompting suffices" conclusion.
- **The gap is untouched by every existing training recipe.** On T2, π_B, SYN64, CAW and CAW-v2 all score E2 = 0. The gap therefore is not specific to CAW training. Training changes only how much the models check.
- **Recommendation: GO for stage 1 of the matched SFT pilot** (CF_s0 and CTRL_s0 with the pre-registered futility gate; ≈ $12, cap $14). Stage 1 needs separate authorization and a top-up: the balance is $4.07.
- **Limits:**
  - one model family;
  - four pre-registered prompts, all given in the system message;
  - greedy decoding.

  "Prompting does not close the gap" holds for these prompts and this 4B model, not for prompting in general or for larger models.
