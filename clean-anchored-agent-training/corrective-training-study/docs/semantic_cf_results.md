# Semantic-counterfactual evaluation: results (2026-11-22)
**Protocol:** `semantic_counterfactual_protocol.md`, frozen at commit `4138bd7` (edit log 1). No changes after inference.

**Evidence:**
- **Archive:** `pod_backups/semcf_eval_20261122.tgz`, sha256 `2c6dc99b6c6483ced82c6fd0e21b1468ab478c4d841e852d47a38a2840a8fec2`. The checksum is identical on the pod and locally. The archive holds per-episode traces, logs, the run script and the exact code tarball.
- **Analysis:** `data/semcf_eval_v0/analysis.{json,md}`, recomputed locally from the archive. It is identical to the pod's output.

**Run facts:**
- **Hardware and cost:** one H100 SXM, 401 s, about **$0.49**.
- **Pre-inference checks:** the spec hash matched the freeze (`a591d07a…`); 22/22 tests passed on the pod, including SYSTEM equality with the training prompt.
- **Runs:** 9 (π_B, prompted π_B, SYN64 ×3, CAW ×2, CAW-v2 ×2), each with 1,920 episodes on **20 held-out API families**.
- **Not done:** no training, no prompt changes, no reruns.

## 1. Results by model
Pooled values are per-family means over the arm's seeds, then the mean over 20 family clusters. The 95% CI is a family-cluster bootstrap.

| metric | π_B | π_B + prompt | SYN64 | CAW | **CAW-v2** | scripted always-check | contract policy |
|---|---|---|---|---|---|---|---|
| initial decision correct | 0.600 | 0.600 | 0.602 | 0.969 | **0.969** [0.960, 0.977] | 1.000 | 1.000 |
| check rate after a fault | 0.000 | 0.000 | 0.003 | 0.741 | **0.745** | 1.000 | 0.200 |
| post-check interpretation | — | — | — | 0.767 | **0.765** [0.756, 0.773] | 0.800 | 1.000 |
| outcome-twin correct (timeouts) | 0.500 | 0.500 | 0.503 | 0.744 | **0.738** [0.724, 0.748] | 0.750 | 1.000 |
| final-state success, feasible cells | 0.875 | 0.875 | 0.876 | 0.997 | **0.993** [0.987, 0.999] | 1.000 | 1.000 |
| **M3** safe escalation, unresolvable cells | 0.000 | 0.000 | 0.000 | 0.000 | **0.003** [0.000, 0.009] | 0.000 | 1.000 |
| **M1** effect sensitivity (A pairs) | 0.000 | 0.000 | 0.000 | 0.000 | **0.003** [0.000, 0.009] | 0.000 | 1.000 |
| **M2** twin joint, history verifier | 0.000 | 0.000 | 0.021 | 0.988 | **0.944** [0.887, 0.988] | 1.000 | 1.000 |
| twin joint, listing verifier | 0.000 | 0.000 | 0.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| B name consistency | 1.000 | 1.000 | 0.994 | 0.911 | **0.877** [0.845, 0.907] | 1.000 | 1.000 |
| premature escalation | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 |
| duplicate-effect rate (all episodes) | 0.167 | 0.167 | 0.166 | 0.083 | **0.083** | 0.083 | 0.000 |
| unnecessary-check rate | 0.000 | 0.000 | 0.001 | 0.541 | **0.545** [0.525, 0.565] | 0.800 | 0.000 |
| excess tool calls over the oracle minimum | −0.08 | −0.08 | −0.08 | +0.37 | **+0.38** | +0.58 | 0.00 |
| fault-free success | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Negative excess calls mean the policy skipped a check the oracle needs.

**Per seed** (full table in `analysis.md`):

| metric | CAW-v2 s0 | CAW-v2 s1 | CAW s0 | CAW s1 |
|---|---|---|---|---|
| M1 effect sensitivity | 0.000 | 0.006 | 0.000 | 0.000 |
| M2 twin joint, history | 1.000 | 0.887 | 1.000 | 0.975 |
| feasible final-state success | 1.000 | 0.987 | 1.000 | 0.994 |
| unnecessary checks | 0.565 | 0.525 | 0.522 | 0.559 |

- **SYN64:** the three seeds are within ±0.01 of one another on every metric.
- **The prompt arm:** 4 of 1,920 completions differ from π_B. The hint was present and had no measurable effect.

**Where CAW-v2 checks:**
- 1.00 after timeouts with any real read tool, including the documented-stale snapshot;
- 0.62–0.74 after timeouts when the only read tool is unrelated (`get_<domain>_office_hours`);
- 0.03–0.28 after a clean 503;
- 0.00 when there is no fault.

**In the unresolvable cells** (non-idempotent write, no decisive verifier), CAW-v2 re-issued the write in 638 of 640 episodes and escalated in 2. CAW re-issued in 640 of 640.

## 2. Gates (protocol §6, applied mechanically by `semantic_cf_analysis.py`)
| step | value | result |
|---|---|---|
| competence precondition (fault-free ≥ 0.85, twin_joint_listing ≥ 0.6) | 1.000, 1.000 | holds |
| H_s falsified (M1 ≥ 0.70 with CI lower bound > 0.5; M2 ≥ 0.70; M3 ≥ 0.70) | M1 0.003, M2 0.944, M3 0.003 | **no** |
| H_s supported (M1 CI upper < 0.5 **or** M2 CI upper < 0.5) | M1 upper bound 0.009 | **yes**, via M1 |
| prompt closes the gap (prompted M1 ≥ 0.70 **and** M2 ≥ 0.70) | 0.000, 0.000 | **no** |
| **pre-registered decision** | | **PROCEED TO MATCHED SFT PILOT** (requires new authorization; not started) |

## 3. Answer to the question
**CAW-v2 learned a verification habit with faithful reading of decisive evidence. It did not learn a conditional recovery policy.**

- **It is the same policy as a scripted always-check rule, at lower cost.** CAW-v2 is within 0.06 of the scripted always-check policy on every correctness metric. It checks on cue after timeouts, reads the result correctly, and completes 99% of completable tasks.
- **It ignores whether repeating is safe (M1 = 0.003)** and whether the read can decide. It checks a stale snapshot, or even unrelated office hours, then re-issues a non-repeat-safe write. In the cases with no safe way to complete, that is unsafe in the applied twin; escalating would have been correct. M3 = 0.003.
- **Verifier interpretation is not the gap.** That part of the shortcut hypothesis is **falsified**. CAW-v2 correctly treats "Error: no changes recorded" as evidence that the write was lost: M2 = 0.944 (CAW 0.988; CAW-v2 − CAW = −0.044 [−0.087, −0.013], driven by seed 1).
- **Renaming the write tool** changes CAW-v2's action in 12% of pairs (B consistency 0.877). CAW-v2 is also slightly *less* invariant to names than the untrained arms, which never vary because they always retry.
- **The pre-registered prompt does not solve the gap.** π_B ignores the added sentence almost completely. Only this one frozen sentence was tested, so stronger prompting strategies are untested: no prompt tuning was allowed.
- **The gap is not created by CAW training.** π_B and SYN64 also score 0 on M1 and M3, because they blindly retry. CAW training replaced blind retrying with blind checking. That fixes the decisive-verifier cases (+0.117 final-state success vs SYN64, [0.110, 0.123]) and halves duplicate effects (0.167 → 0.083). It leaves the side-effect and observability conditions untouched.

## 4. Limits
1. **The CIs are narrow because behaviour is nearly constant across families, not because the sample is large.**
   - All 20 families share one generator template, so family clustering removes episode correlation but not design homogeneity.
   - The dominant uncertainty is **external validity** (other API designs, real services); the CIs do not measure it.
2. **Two seeds for CAW / CAW-v2.** Seed-level differences are visible: M2 1.000 vs 0.887. All gate-relevant conclusions hold in **every** seed of every arm.
3. **One model family** (Qwen3-4B-Instruct-2507), **one** prompt-only variant, greedy decoding.
4. **The contract policy is an upper bound by construction:** it parses the known templates. It shows that the information is present and sufficient, not that it is easy to extract.
5. **"Escalate" is the only credited action in unresolvable cells.** No model ever produced a non-escalating safe alternative, such as a text reply asking the user: 0 such cases were found when re-issues were excluded. So this strictness did not affect any arm.

## 5. What follows
- **The pilot is not started.** The pre-registered next step is the matched SFT pilot (protocol §7). It **needs new authorization**: estimated $25–35, against a current RunPod balance of **$5.88**.
- **What the pilot would test.** The pilot asks whether balanced counterfactual SFT on the 20 *train* families teaches effect- and observability-conditioned recovery that transfers to the held-out families and held-out combinations. The comparison is a same-size check-first-only control.
- **What the result already supports, without any pilot:** an **empirical-generalization paper (type B).** Execution-verified recovery SFT transfers a check habit across environments, but not side-effect semantics; neither the base policy nor a contract-pointing prompt supplies them.
