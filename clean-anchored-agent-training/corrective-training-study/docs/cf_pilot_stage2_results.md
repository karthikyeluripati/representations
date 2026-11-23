# CF-SFT pilot, stage 2: final gates (2026-11-23)

**Protocol:** `cf_sft_pilot_protocol.md`, frozen at `657f44a`. The training data, hyperparameters, evaluation sets and gates were not changed. The gates were applied mechanically by `scripts/cf_pilot_analysis.py --stage final`.

## Decision: NO CLAIM

The frozen decision map stops at its first rule: **S4 (no harm) failed**. **S5 (mechanism) failed as well**, and on its own that would also have given NO CLAIM ("gain not attributable to contract text").

| gate | criterion | result | outcome |
|---|---|---|---|
| S1 | CF − CTRL on E2 ≥ 0.30, CI > 0, every seed | +0.507 [0.494, 0.520]; every CF seed above every CTRL seed | **pass** |
| S2 | CF − CTRL on E1 ≥ 0.20, CI > 0, every seed | +0.234 [0.225, 0.243] | **pass** |
| S3 | CF E3 ≥ 0.50 and CF − CAW-v2 ≥ 0.30 | E3 = 0.069 [0.035, 0.104] | **fail** |
| S4 | no harm (9 checks) | 8 pass; **τ² check rate 0.590 vs CAW-v2 0.846** (threshold 0.746) | **fail** |
| S5 | CF − ABL on E2 ≥ 0.20 | **+0.003 [−0.010, 0.015]** | **fail** |
| S6 | prompt baseline within 0.05 of CF | π_B@P1 and CAW-v2@P0: E2 0, E1 0.50, E3 0 | does not hold |

**Run facts:**
- **Jobs:** all ten Stage-2 jobs completed with no failures. These were six trainings (CF s1–s2, CTRL s1–s2, ABL s0–s1) and four regression runs (CF s0–s2, CTRL s0).
- **Hardware and cost:** one H100, 7.4 h, **$32.77** (cap $58). Stage-1 CF_s0 and CTRL_s0 were reused unchanged; their adapter checksums match.
- **Guard:** the fixed guard (`a1c877d`) detected completion within one poll, verified the archive, stopped the pod and confirmed a storage-only spend rate.
- **Archive:** `pod_backups/cf_pilot_stage2_20261123.tgz`, sha256 `ca0024e9e345b250e6e4dce22b926cb161d780b21e52263fcbc8e981fc3d8d4c`, identical on the pod and locally. It holds 49,920 diagnostic episodes, the regression traces, the logs, the configs and `ADAPTER_SHA256SUMS`. The adapters stay on volume `9ghq6ztmfl`.
- **Outputs:** `data/cf_pilot_stage2/final.json` (gates) and `data/cf_pilot_stage2/diagnostics.json` (descriptive, below).

## 1. Every seed (T2-test: 20 held-out families, 4,320 episodes per run; T1: frozen, 1,920 episodes)
| run | E2 | E1 | safe escalation | reconcile | feasible | fault-free | unnecessary checks | T1 E3 | T1 feasible |
|---|---|---|---|---|---|---|---|---|---|
| CF_s0 | 0.481 | 0.722 | 0.125 | 0.944 | 0.961 | 0.996 | 0.021 | 0.087 | 0.989 |
| CF_s1 | 0.528 | 0.731 | 0.300 | 0.956 | 0.967 | 1.000 | 0.048 | 0.031 | 0.992 |
| CF_s2 | 0.512 | 0.750 | 0.087 | 1.000 | 0.980 | 1.000 | 0.048 | 0.087 | 0.995 |
| ABL_s0 | 0.503 | 0.750 | 0.031 | 1.000 | 0.995 | 1.000 | 0.100 | 0.009 | 0.998 |
| ABL_s1 | 0.506 | 0.653 | 0.478 | 1.000 | 0.928 | 0.999 | 0.127 | 0.050 | 0.932 |
| CTRL_s0–s2 | 0.000 (each) | 0.500 | 0.000 | 0.000 | 0.951 | 1.000 | 0.64 | 0.000 | 0.996–1.000 |
| CAW-v2 s0–s1 | 0.000 | 0.497–0.500 | 0.000 | 0.000 | 0.932–0.942 | 0.997–1.000 | 0.64 | 0.000 | 0.985–1.000 |

**Regression, per seed** (S4 compares the CF pool against the CAW-v2 pool):

| run | BFCL S0 test | BFCL S0 held-out | BFCL held-out-class implicit | τ² check | τ² end-state |
|---|---|---|---|---|---|
| CF_s0 | 0.814 | 0.632 | 0.743 | 0.615 | 0.115 |
| CF_s1 | 0.860 | 0.640 | 0.871 | 0.538 | 0.192 |
| CF_s2 | 0.791 | 0.614 | 0.686 | 0.615 | 0.192 |
| CTRL_s0 | 0.837 | 0.632 | 0.771 | 0.385 | 0.115 |
| CAW-v2 s0 / s1 | 0.814 / 0.860 | 0.649 / 0.632 | 0.743 / 0.757 | 0.692 / 1.000 | 0.231 / 0.192 |

τ² is the diagnostic airline subset. Its event counts are small: one event is about 0.04 on end-state.

## 2. Why E2 plateaus at 0.5 for both CF and ABL (descriptive; not a gate)
T2 hides the retry contract only in the tool description. When the contract is stripped, idempotent and non-idempotent tools look identical at decision time. A contract-blind policy can still act correctly in many cells:
- **keyed:** the `request_key` parameter is visible in the schema;
- **a reverse tool is present:** "re-issue, read the receipt, reverse any duplicate" is correct for every effect. The receipt shows the effect after the fact.

The only cells where reading the contract is necessary are **non-decisive verifier with no reverse tool**. There the idempotent tool should be re-issued, and the non-idempotent one should be escalated without repeating the write.

| E2, split by compensation | CF s0 / s1 / s2 | ABL s0 / s1 | CTRL, CAW-v2 |
|---|---|---|---|
| reverse tool present (no contract needed) | 0.944 / 0.956 / 1.000 | 1.000 / 1.000 | 0 |
| **reverse tool absent (contract needed)** | **0.019 / 0.100 / 0.025** | **0.006 / 0.013** | 0 |

In those contract-needed cells, actions are **the same for idempotent and non-idempotent tools**. Counts below are out of 320 per effect:

| run | idempotent: re-issue / escalate | non-idempotent: re-issue / escalate |
|---|---|---|
| CF_s0 | 282 / 38 | 280 / 40 |
| CF_s1 | 242 / 78 | 224 / 96 |
| CF_s2 | 297 / 23 | 292 / 28 |
| ABL_s1 | 172 / 148 | 167 / 153 (it also escalates 62 keyed episodes) |

**What this shows:**
- **The models do not read the contract sentence.** Escalation happens at a roughly constant rate, independent of effect. The non-zero "safe escalation" scores, including ABL_s1's 0.478, come from that effect-blind escalation rate, not from discrimination.
- **The pre-registered E2 has a contract-free ceiling of about 0.5.** That is exactly where both CF and ABL sit. The futility threshold (0.40) and S1 (+0.30 over CTRL) were therefore reachable without contract reading. This flaw is in the pilot design and was not recognised at pre-registration. S5, the ablation, is the gate that exposed it, which is what it was for. Nothing was re-scored.

## 3. What ordinary counterfactual SFT solved and what it does not generalize
| | evidence | depends on the contract? |
|---|---|---|
| **Solved:** reconcile through a reverse tool after an ambiguous timeout | reconcile 0.97 vs 0 for CTRL and CAW-v2, every seed, on held-out families | **no**; ABL matches it (1.00) |
| **Solved:** same-key retry | 1.00 (CTRL also 1.00) | no; visible in the schema |
| **Solved:** removes the always-check habit at no cost | unnecessary checks 0.04 vs 0.64; feasible 0.969 vs 0.951; fault-free 0.999 | no |
| **Solved:** no damage to ordinary BFCL tool use | S0 test and held-out and held-out-class implicit all within the S4 margins of CAW-v2 | — |
| **Not learned:** telling non-repeat-safe from repeat-safe writes from the contract text | E2 with no reverse tool ≤ 0.10, every seed; identical action distributions across effects; CF − ABL = +0.003 | — |
| **Not learned:** escalating when the outcome cannot be resolved | safe escalation 0.09–0.30, effect-blind; 0 on the held-out stale-verifier combination | — |
| **Not generalized:** to a structurally different template | T1 E3 0.069 [0.035, 0.104] vs the 0.50 bar | — |
| **Regressed:** τ² check-after-timeout vs CAW-v2 | 0.59 vs 0.85. End-state 0.167 vs 0.212 is within the margin, but lower | — |

## 4. Limits
- **Model:** one model family (Qwen3-4B-Instruct-2507).
- **Seeds:** 3 seeds (2 for ABL).
- **Templates:** synthetic. T2 shares one template across families, so T1 is the only cross-template check.
- **τ²:** the subset is small.
- **Scope:** the negative result concerns this data recipe, at this size, on this model. It does not show that contract reading cannot be learned.

No further training, objective or thesis change was made. Anything further needs separate authorization.
