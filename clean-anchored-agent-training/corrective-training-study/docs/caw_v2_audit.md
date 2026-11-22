# CAW-v2 audit (2026-11-22)

**Scope.**
- **Code and data:** branch `research/semantic-counterfactual`, from `uof-benchmark` @ `8f7add6`.
- **Inputs:** every number below was recomputed from the archived per-episode traces in `pod_backups/` (sha256 prefixes in `RELEASE.md`), with `scripts/audit_caw2.py`. Outputs are in `data/audit_caw2/{audit,audit_supp}.json`.
- **What is not changed:** no pre-registration, gate or archived result.

**Where this audit disagrees with the earlier write-up** (`method_caw2.md`, my summary to the user), it says so explicitly.

## 1. What CAW-v2 is
| | SYN64 | CAW | CAW-v2 |
|---|---|---|---|
| synthetic part (832 records, 64 classes) | verb-keyed: retry idempotent writes, check non-idempotent ones | **check after a timeout on every write** (create/set/remove), then branch on the observation; 503 and reads: retry | identical to CAW |
| BFCL part (517) | `iar_v1`, including `idem_retry` records | `iar_v1` **minus `idem_retry`**, cycled | CAW's part minus 88, **plus 88 `persist_backup` / `persist_escalate`** |
| algorithm | LoRA SFT from π_B (r8, lr 1e-4, 3 epochs) + 84 anchors | same | same |

**Algorithm: ordinary supervised fine-tuning.**
- No new objective, no DPO and no RL.
- π_B itself is π₀ + a CSO-style verified DPO adapter; that is the only preference step, and it predates CAW.

**The rule is a design choice, not something the model discovered.** `build_caw_data.py` says it outright: "No idempotency inference is needed". CAW-v2's training set contains:
- **no** record where re-issuing a write directly after a timeout is correct;
- **no** record without a verifier;
- **no** record where the verifier is inconclusive (indecisive classes are dropped);
- **no** record where the verifier's "lost" evidence is error-shaped.

So the question "did CAW-v2 learn a universal check-after-uncertain-write heuristic?" has an answer by construction: **yes, that is the rule it was trained on.** The open questions are narrower:
- (d) does it *interpret* the check?
- Does it do anything sensible where the rule does not apply: no verifier, an inconclusive verifier, or a side effect that changes the right action?

The existing evaluations cannot answer the second question (section 5).

## 2. Re-computation from raw traces
| reported finding | recomputed | status |
|---|---|---|
| τ² implicit check rate 0.846 vs SYN64 0.026 | 0.846 vs 0.026; +0.821 [0.654, 0.962] (13-event bootstrap) | **verified from raw traces** |
| τ² cancellations checked 38/44 | 38/44 (CAW 28/44, SYN64 0/66) | **verified** |
| BFCL held-out-class (Travel) implicit 0.750 | 0.750 (CAW 0.829) | **verified** |
| … vs SYN64 0.667, +0.083 [0.002, 0.155] | SYN64's BFCL traces are **not** in any local archive; only derived per-task values (`analysis_caw2.json`) exist | **reported only** (the raw traces are on the network volume) |
| S1\|S0 0.95 / 0.99 | 0.95 / 0.99 | **verified** |
| original test 0.878, held-out 0.727 (composite "overall") | 0.856 / 0.900 and 0.728 / 0.726 per seed | **verified** |
| B reference 0.872 / 0.732 | B_s0 0.872 / 0.732 (B_s1 0.858 / 0.730; B_s2 0.867 / 0.731) | **verified** |
| impossible-request suite (S3) test 1.00 | 1.000 / 1.000 (B 0.930) | **verified** |

**Pre-registered rule.** It was applied as written: (i)–(iv) hold, except that (ii) rests on reported-only SYN64 numbers. The claim **stands under its own rule**. The points below are additional weaknesses; they are not reinterpretations of the gate.

## 3. Weaknesses the gate does not capture
1. **Fault-free competence regressed. The composite masks it.**
   - Test S0, clustered by 22 base tasks: CAW-v2 0.791 vs B 0.841 (3 seeds), **−0.050 [−0.108, −0.005]**.
   - Held-out S0, 49 tasks: **−0.055 [−0.130, +0.015]**.
   - Rule (iv) passed only because S3 escalation (1.00) and lower false-outage rates offset the S0 loss inside "overall".
   - Held-out *premature abstention* rose from 0.07 (B) to 0.11.
2. **τ² end-state success did not improve significantly.** The check rate is not recovery.

   | τ² diagnostic, implicit (13 events / 8 tasks) | SYN64 | CAW | CAW-v2 |
   |---|---|---|---|
   | check rate | 0.026 | 0.654 | 0.846 |
   | fault-local decision correct, applied twin (no re-issue) | 0.000 | 0.615 | 0.808 |
   | fault-local decision correct, lost twin (re-issue) | 1.000 | 0.731 | 0.692 |
   | twin discrimination (both decisions right) | 0.000 | 0.346 | 0.500 |
   | **end-state success** | 0.154 | 0.250 | **0.212** |
   | twin joint end-state success | 0.000 | 0.115 | 0.154 |

   - CAW-v2 − SYN64 end-state: **+0.058 [−0.038, 0.173]** (event clusters); +0.031 [−0.094, 0.188] (task clusters).
   - S0 is 0.10–0.18, so most episodes fail for competence reasons before recovery matters.
   - On the S0-solved events, CAW-v2 succeeds 8/8 vs SYN64 6/12. That is suggestive only: just 2 events qualify.
3. **Some of CAW-v2's τ² "lost" episodes now escalate** (escalation 0.135 vs 0 for CAW), so the lost-twin decision drops to 0.69.
   - On τ² *idempotent control* writes, CAW-v2's end-state success is −0.110 [−0.222, −0.026] vs SYN64 (21 task clusters).
   - The pre-registration called the extra checks there "cheaper" (4.33 vs 5.64 calls). Fewer calls partly reflect more escalation.
4. **CAW-v2 is worse than CAW on BFCL held-out:** −0.079 [−0.150, −0.007] success and −0.129 [−0.257, 0.000] twin discrimination. CAW-v2 also scores 0.75 vs 0.875 on the 6 in-distribution decisive tasks.
5. **The τ² evaluation design favours checking.**
   - Every τ² diagnostic event is a non-idempotent write with a decisive read, by selection.
   - The forced gold prefix already contains the read call: **26 of CAW-v2's 44 checks are byte-identical repeats of a call in the prefix.**
   - The `checked()` metric counts any `get_*` call. 38/44 are the relevant read (the right reservation or user), so this metric issue is minor.
6. **Wording overlap.** The test wording "Error: the request timed out." is lexically unseen, but training contains "gateway timeout" and "connection dropped". It is therefore held out lexically, not semantically.
7. **Leakage.** No TravelAPI task appears in any CAW/CAW-v2 BFCL record (checked by fault class). However, 24 of the 35 held-out-class tasks are train-split tasks that π₀ saw fault-free. That task familiarity is shared by every arm compared.

## 4. Uncertainty
- **What the CIs cover.** All CIs are paired cluster bootstraps over tasks (BFCL) or events (τ²). Seeds are averaged inside a cluster, so the CIs are **conditional on the trained seeds** and exclude training-seed variance.
- **Two seeds** cannot estimate that variance. The per-seed spread is large: τ² check rate 0.692 vs 1.000, and twin discrimination 0.385 vs 0.615.
- **τ² events share tasks** (13 events from 8 tasks). Task-cluster intervals are wider and are reported next to the event intervals in `audit.json`.
- **Repeated wordings and twins are not independent units.** The audit never counts them as such: they are averaged within a cluster.

## 5. Does a universal check-first policy explain the gains?
- **Mostly yes, for the check rate.** No evaluation item in either environment penalises checking, beyond its cost:
  - BFCL decisive tasks and τ² diagnostic events are all non-idempotent writes with decisive verifiers;
  - S1 and τ² controls tolerate an extra read.
  - A policy that always checks scores at the ceiling on every "check" metric used in the paper.
- **But the model does read the observation.** After a check, CAW-v2 re-issues the write:
  - τ²: in 3% of applied twins vs 65–69% of lost twins;
  - BFCL held-out: in 17–26% vs 91–93%.
- **Interpretation is imperfect and not better than SYN32's.**
  - The 17–35% re-issues after an "applied" observation on BFCL are duplicate writes made *after seeing the evidence*.
  - CAW's interpretation is better than CAW-v2's.

**Summary:**
- CAW-v2 learned a **portable check reflex plus partial observation-conditioning**.
- Whether it conditions on side effects or observability cannot be determined: the existing benchmarks never make those factors decision-relevant. Section 1 explains why the training data cannot have taught it this either.

## 6. Correction to my earlier report
I previously told you all four "main-track signal" conditions were met. They were, as pre-committed. But that summary left out three things:
- the non-significant τ² end-state effect;
- the masked fault-free regression;
- three directly relevant prior works, found in this audit (`research_thesis_next.md` §3).

A main-track *method* claim is not supported by the current evidence.
