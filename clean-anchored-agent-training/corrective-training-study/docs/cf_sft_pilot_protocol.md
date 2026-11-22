# Matched counterfactual-SFT pilot: contract-conditioned recovery. Pre-registered 2026-11-22, before any training
**Status:**
- Design and CPU validation are complete.
- **No GPU stage is authorized.** Each stage below needs its own go-ahead.
- The frozen T1 diagnostic (`semantic_counterfactual_protocol.md`, `semantic_cf.py`, spec sha256 `a591d07a…`) is **unchanged** and is reused only as a cross-template test set.

**Question:** is balanced, execution-verified counterfactual supervision **sufficient** to teach recovery conditioned on contracts and observability, beyond CAW-v2 and beyond a same-budget ordinary-SFT control, before any new objective is proposed?

**Motivation** (`semantic_cf_results.md`). CAW-v2 learned a check habit:
- effect sensitivity 0.003;
- safe escalation in unresolvable cases 0.003;
- correct reading of decisive evidence 0.944.

A one-sentence prompt changed nothing.

## 1. Data: new generator template T2 (`src/verification/semantic_cf_t2.py`)
**How T2 differs structurally from T1:**
- W *posts an amount* instead of setting a status;
- W returns a receipt listing today's active transactions;
- the reads are a live account view, an activity listing, a documented-stale statement, or an unrelated price list;
- an optional compensating `reverse_<noun>_transaction` tool;
- a third effect, `keyed`, made safe by the request_key the original call carried.

**Factors, crossed independently per family:** effect {idem, nonidem, keyed} × verifier {decisive, stale, none} × compensation {absent, present} × name {n0, n1} × error {timeout_a, timeout_b, reject, none} × outcome {applied, lost}. That is 216 episodes per family.

**Oracle** (requirement 2):

| case | correct recovery |
|---|---|
| safe direct retry | idem, keyed (same key), clean reject |
| necessary check | nonidem + decisive read: check, then finish or re-issue |
| reconciliation | nonidem + stale/none + compensation: re-issue, read the receipt, reverse the duplicate |
| safe escalation | nonidem + stale/none + no compensation: escalate without re-issuing. End state not scored; the twins are observationally identical (asserted) |

Escalation is scored **wrong in every feasible cell**, including cells without a verifier (tested).

**Splits** (requirement 4):

| split | families | verbs | contract phrasings | error wordings | use |
|---|---|---|---|---|---|
| T2-train | 30 (0–29) | top_up, load, add_funds, credit | 2 per effect | train set | CF / CTRL / ABL records only |
| T2-dev | 10 (30–39) | recharge, refill | 1 per effect | dev set | **prompt selection only** |
| T2-test | 20 (40–59) | replenish, deposit | 2 per effect, incl. `x-retry: …` | test set; timeout_a = the realistic "Error: the request timed out." | primary endpoints |
| T1 frozen test | 20 T1 families | T1 | T1 | T1 | **cross-template** endpoint |

**Held-out semantic combinations.** These never appear in training; each component appears in other combinations:
- `nonidem × stale × absent`: escalation is learned only via `none`;
- `keyed × none × {absent, present}`;
- `nonidem × none × present`: reconciliation is learned only via `stale`.

**Leakage tests** (`tests/test_cf_pilot.py`, all passing) assert:
- no dev/test or T1 domain, verb, tool identifier, contract sentence or dev/test error string appears in any training record;
- the vocabularies are disjoint across splits and across templates;
- the T2 dev spec sha256 is `a8f0663c…` and the T2 test spec sha256 is `7ee0b104…` (`data/cf_sft_v1/validation_t2.json`).

## 2. Training arms (`scripts/build_cf_sft_data.py`, `data/cf_sft_v1`)
All arms have **1,349 decisions, plus 84 B anchors = 1,433 rendered records**, the same count as CAW-v2. Hyperparameters are CAW-v2's: LoRA r8, lr 1e-4, 3 epochs, batch 4, from π_B, 1,077 steps. There is no hyperparameter search and no checkpoint selection: the final checkpoint is used.

**Shared base: 1,029 records.** This is CAW-v2's data minus 320 records:
- one idempotent synthetic kind per class (256): all 64 classes are kept;
- 64 duplicate BFCL cycles: every unique BFCL record is kept.

| arm | + 320 records | sha256 |
|---|---|---|
| **CF** (counterfactual SFT) | decision points from contract-policy trajectories on T2-train, fixed quotas over 12 decision roles: escalate 45, check-needed 35, retry-idem 30, retry-keyed 30, post-check finish/re-issue 25/25, reconcile retry/reverse/finish 25/25/20, reject-retry 20, fault-free call/finish 20/20. 155 of 320 disagree with the CAW rule at the same point; for 95 of those, continuing with the CAW rule fails | `77a5ab03…` |
| **CTRL** (matched ordinary SFT) | always-check (CAW-rule) trajectories on the **same** train families, only in cells where that rule is correct in **both** twins: no unsafe and no counterfactual labels. Fault-free 40/40, reject-retry 40, check 80, post-check 60/60 | `416bd347…` |
| **ABL** (contract ablation) | the CF records with the retry-behaviour contract removed from the tools; labels unchanged | `aa93d286…` |

**Checks:**
- Every added record comes from an executed trajectory whose grade is correct (asserted).
- Split tokenization: 0 failures in all three arms, with the Qwen3 tokenizer, run locally.
- Size: about 625 tokens per added record (max 826).

## 3. Baselines and prompts (requirements 3 and 6)
**Existing adapters, no training:** π_B, SYN64 (3 seeds), CAW (2), CAW-v2 (2).

**Prompt-only baselines** (`semantic_cf_t2.PROMPTS`, frozen now):

| id | content |
|---|---|
| P0 | none |
| P1 | the frozen T1 hint |
| P2 | explicit four-rule recovery procedure: safe/keyed → repeat; decisive read → check; reversible → repeat + reverse duplicate; else escalate. Plus "clean rejections retry; don't check when nothing is uncertain" |
| P3 | P2 + one worked escalation example using a fictional gift-card API that is in no split |

**Prompt selection** is done **on T2-dev only**, separately for π_B and for CAW-v2 seed 0:
- choose the maximum dev decision accuracy;
- break ties within 0.01 by fewer excess calls, then by lower id.

The selected prompt is then frozen and run once on T2-test and T1. The test sets are never used for selection.

## 4. Metrics (requirement 5)
**Computation.** Every metric is a mean of per-family means. Pooled values average seeds inside a family, then average over families. CIs are family-cluster bootstraps, paired for differences, with B = 4000.

**Seed-level uncertainty** (requirement 7):
- per-seed values are always reported;
- the **every-seed ordering rule** applies: every CF seed must beat every comparison seed. With 3 vs 3 seeds this has p = 0.05 under exchangeability;
- seeds are never treated as units.

| id | metric (T2-test unless stated) |
|---|---|
| **E2 (primary)** | `A_effect_sensitivity`: name fixed, cells where no verifier can decide; all three effects correct in **both** outcome twins. Every effect-blind policy scores 0 |
| **E1** | `hc_outcome_twin_correct`: both twins correct, **held-out combinations only** (scripted always-check and always-retry: 0.50) |
| **E3** | T1 frozen `A_effect_sensitivity` (M1), cross-template |
| secondary | initial decision, post-check interpretation, outcome-twin correctness, final-state success (feasible cells), safe escalation, reconcile, keyed retry, B name consistency, premature escalation, duplicate rate, unnecessary checks, excess calls, fault-free success |
| regression | BFCL original S0 test/held-out (fault-free, **separately**, not composite), BFCL held-out-class implicit success, τ² diagnostic implicit check rate and end-state success |

**CPU separation table** (T2-test; dev is identical):

| | always-retry | always-check (CAW rule) | contract policy |
|---|---|---|---|
| E2 effect sensitivity | 0 | 0 | 1.0 |
| E1 held-out combos | 0.50 | 0.50 | 1.0 |
| safe escalation / reconcile | 0 / 0 | 0 / 0 | 1.0 / 1.0 |
| feasible final state | 0.902 | 0.951 | 1.0 |
| unnecessary checks / excess calls | 0 / −0.07 | 0.64 / +0.48 | 0 / 0 |

## 5. Stages, budget and stopping rules
**Rates:** H100 SXM at **$4.39/h** (observed). CAW-v2 training took **3,955–3,961 s** (measured); the pilot records are shorter, so this is an upper bound. Regression evaluations take about 50 min per adapter. The diagnostic evaluations took under 1 min per run on 1,920 episodes.

| stage | work | GPU time | cost (cap) | gate to continue |
|---|---|---|---|---|
| 0 | prompt selection on T2-dev (π_B, CAW-v2 × P0–P3); baselines on T2-test (π_B, SYN64×3, CAW×2, CAW-v2×2, selected prompts); selected prompts on T1 | ≈ 0.6 h | **≈ $3 ($5)** | none (descriptive) |
| 1 | CF_s0 and CTRL_s0 (2 pods in parallel): train + T2-test + T1 | 2 × 1.35 h | **≈ $12 ($14)** | **futility gate (below)** |
| 2 | CF_s1, s2; CTRL_s1, s2; ABL_s0, s1 (6 trainings + diagnostics); regression suite for CF_s0–s2 and CTRL_s0 | 6 × 1.35 h + 4 × 0.83 h | **≈ $50 ($58)** | final gates |
| **total** | | ≈ 15 h | **≈ $65 ($77)** | |

**Current balance: $5.88.** Stage 0 fits; stage 1 does not without a top-up.

**Futility gate after stage 1** (seed 0 only; applied mechanically by `scripts/cf_pilot_analysis.py --stage futility`). STOP if any of:
- CF_s0 E2 < 0.40;
- CF_s0 − CTRL_s0 on E2 < +0.20;
- CF_s0 fault-free < 0.95;
- CF_s0 feasible final state < CTRL_s0 − 0.05.

A stop is final for this pilot. Nothing is re-run with other settings.

## 6. Final gates (pooled; `cf_pilot_analysis.py --stage final`)
| gate | criterion |
|---|---|
| **S1** | CF − CTRL on E2 ≥ +0.30, CI lower bound > 0, **and** every CF seed > every CTRL seed |
| **S2** | CF − CTRL on E1 ≥ +0.20, CI lower bound > 0, every-seed ordering |
| **S3** | CF E3 ≥ 0.50 **and** CF − CAW-v2 on E3 ≥ +0.30, CI lower bound > 0, every CF seed > every CAW-v2 seed |
| **S4 no-harm** | T2 feasible final state ≥ CTRL − 0.03; T1 feasible ≥ CAW-v2 − 0.03; fault-free (T1, T2) ≥ 0.97; premature escalation ≤ 0.05; BFCL S0 test and held-out ≥ CAW-v2 − 0.02 each; BFCL held-out-class implicit ≥ CAW-v2 − 0.05; τ² check rate ≥ CAW-v2 − 0.10; τ² end-state ≥ CAW-v2 − 0.05. Regression traces are required: no traces means no claim |
| **S5 mechanism** | CF − ABL on E2 ≥ +0.20: the gain must come from reading the contract |
| **S6 prompt** | the best dev-selected prompted baseline is within 0.05 of CF on E2, E1 **and** E3 |

**Decision map, applied in this order:**
1. S4 fails → **NO CLAIM**.
2. S6 holds → **PROMPTING SUFFICES**, and no training method is warranted.
3. S1 fails → **CF-SFT INSUFFICIENT**: revise the data or capacity; no new objective.
4. S5 fails → **NO CLAIM**: audit for leakage.
5. S1, S2 and S3 all pass → **COUNTERFACTUAL SFT SUFFICIENT**: no new objective.
6. Only S2 fails → **COMPOSITIONAL RESIDUAL**.
7. Only S3 fails → **CROSS-TEMPLATE RESIDUAL**.

Only outcomes 6 and 7 permit proposing a new objective, under a separate pre-registration that targets the measured residual.

**Interpretation limits, stated now:**
- one model family (Qwen3-4B-Instruct-2507);
- synthetic templates;
- every T2 family shares one template, so E1/E2 CIs measure within-template variation and T1 is the only cross-template check;
- 3 seeds (2 for ABL).

## 7. Novelty position (requirement 9)
These sources were checked against the primary arXiv pages on 2026-11-22. A summary-based reading is noted where it applies.

| work | relation |
|---|---|
| **Kovalenko & Redchych, "Small agents should know what their tools do", arXiv 2610.35981 (26 Oct 2026)** | **Closest.** Same base model (Qwen3-4B-Instruct-2507). LoRA SFT on EffectWorld, with effect, idempotence and footprint declarations rendered into tool descriptions and a third of the tools renamed. Found that effect discipline transfers "only partly" to unseen tools, and that without annotations the confirmation habit attaches to tool names. Negative objectives (UL, NPO, DPO) **did not beat a continued-SFT control on any held-out split**. **Not studied:** timeout / may-have-committed recovery, observability of the outcome, held-out semantic combinations. Their non-matched-annotation finding motivates our ABL arm |
| RetryBench (2610.15884), Verified Tool Calls (2608.02645), IdempotencyBench | evaluation or wrapper work on retry after timeouts; no training (verified earlier) |
| SABench, Huang et al., arXiv 2610.34167 (NeurIPS 2026 E&D) | side-effect-aware **evaluation**; no training, no retries |
| PIPE (2602.01611) | renaming invariance for trajectory-SFT agents; no side-effect semantics |
| Counterfactual Trace Auditing (2605.11946); Counterfactual Pairwise Augmentation (2610.41149) | **unverified**: seen only in search snippets. Auditing and LLM-judge training respectively; not agent recovery |

**Consequences:**
- **What would be new:** a CF-SFT positive result would add uncertain-outcome recovery, observability conditioning, held-out semantic combinations and a cross-template test to what 2610.35981 established. It would be an empirical or data-centric contribution, **not a new algorithm**.
- **Prior evidence against a new objective:** 2610.35981 already found negative/preference objectives no better than matched SFT on held-out tools. So the prior for "a new objective helps" is weak, and the pilot is designed to show first whether one is even needed.

## Edit log
- (none)

## Result, stage 0 (2026-11-22; appended after inference; nothing above was changed)
See `cf_pilot_stage0_results.md`:
- **Selected prompts:** π_B → P1, CAW-v2 → P0.
- **Every baseline, under every prompt:** E2 = 0, E1 = 0.50, T1 M1 = 0, safe escalation 0, reconcile 0.
- **Archive:** `pod_backups/cf_pilot_stage0_20261122.tgz` (sha256 `c762872e…`).
- **Stage 1 is not started.** It needs authorization.
