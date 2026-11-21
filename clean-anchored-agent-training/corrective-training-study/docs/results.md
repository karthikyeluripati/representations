# Results

**No baseline (A or B) has been trained yet.** This file holds only measurements on the archived base and π₀ traces: original outputs from the 2026-11-19 H100 gate, recomputed exactly by `scripts/check_archive.py`.

## Archived reference evaluations (raw correct / n by suite)
Source: `results/archive_check.json`. Headline values:

| split | policy | fault-free | genuine repair | false-outage rate | overall (macro) |
|---|---|---|---|---|---|
| held-out | base | 0.439 | 0.360 | 0.009 | 0.376 |
| held-out | π₀ | 0.649 | 0.880 | 0.355 | 0.740 |
| test | base | 0.767 | 0.583 | 0.056 | 0.631 |
| test | π₀ | 0.837 | 0.600 | 0.583 | 0.808 |

## Failure taxonomy of π₀ (`results/failure_types/pi0.json`)
Primary label per **failed** episode:

| split | B false outage | C argument | D ordering | G general | E premature escalation | F unneeded retry | X missed escalation | total failed |
|---|---|---|---|---|---|---|---|---|
| held-out | **162** | 72 | 76 | 71 | 11 | 8 | 3 | 403 |
| test | **40** | 44 | 0 | 11 | 5 | 0 | 0 | 100 |

H (interface misuse) is 0 on both splits.

For comparison, the base model's held-out failures are dominated by:
- missed escalation: 334;
- ordering: 286;
- general: 265;
- argument: 170.

Base has only 6 false-outage failures. π₀ removes most competence failures and creates false-outage failures.

**Real-error events** (an agent call returning a real error, followed by the agent's next action), split by whether the error *type* appears in any train corrective state:

| split | policy | seen type: outage response / total | unseen type: outage response / total |
|---|---|---|---|
| held-out | π₀ | 35 / 52 (0.67) | 449 / 1,311 (0.34) |
| test | π₀ | 88 / 169 (0.52) | 38 / 46 (0.83) |
| held-out | base | 0 / 12 | 6 / 389 |
| test | base | 9 / 121 | 1 / 29 |

**Reading.**
- False-outage behaviour is π₀'s largest failure category: 40% of failed held-out episodes and 38% of failed test episodes.
- It is **not specific to unseen error types**. π₀ also answers seen error types with outage responses.
- Repeated retries inflate event counts, so episode-level counts (the first table) are the primary measure.
- Whether π₀ *recognises* the error type cannot be read from the traces, because the only free text is "Done.". This needs the recognition probe (GPU).

## Baselines A and B (H100, 2026-11-19; one seed; π₀ = rebuilt π₀; identical episodes)

**Correct / n by suite:**

| split | policy | S0 | RP | RA | S1 | S2 | S3 |
|---|---|---|---|---|---|---|---|
| held-out | π₀ (rebuilt) | 71/114 | 71/81 | 17/19 | 225/342 | 217/342 | 338/342 |
| held-out | A: balanced SFT | 79/114 | 78/81 | 16/19 | 235/342 | 224/342 | 342/342 |
| held-out | B: verified DPO | 77/114 | 72/81 | 15/19 | 234/342 | 209/342 | 339/342 |
| test | π₀ | 36/43 | 5/7 | 36/53 | 105/129 | 102/129 | 129/129 |
| test | A | 38/43 | 6/7 | 53/53 | 111/129 | 110/129 | 129/129 |
| test | B | 37/43 | 6/7 | 52/53 | 106/129 | 104/129 | 127/129 |

**Headline metrics:**

| split | policy | overall | fault-free | genuine repair | false-outage rate | premature abstention |
|---|---|---|---|---|---|---|
| held-out | π₀ | 0.731 | 0.623 | 0.880 | 0.305 | 0.146 |
| held-out | A | **0.763** | 0.693 | **0.940** | 0.186 | 0.121 |
| held-out | B | 0.732 | 0.675 | 0.870 | **0.110** | 0.090 |
| test | π₀ | 0.809 | 0.837 | 0.683 | 0.585 | 0.096 |
| test | A | **0.893** | 0.884 | **0.983** | **0.364** | 0.024 |
| test | B | 0.867 | 0.860 | 0.967 | 0.575 | 0.063 |

**Paired differences (cluster bootstrap by base task, 95% CI):**

| comparison | split | overall | genuine repair | false-outage rate | other |
|---|---|---|---|---|---|
| A − π₀ | held-out | +0.032 [+0.001, +0.063] | +0.06 [0.00, +0.13] | −0.118 [−0.205, −0.020] | fault-free +0.07 [+0.016, +0.135] |
| B − π₀ | held-out | +0.001 [−0.030, +0.030] | −0.01 [−0.055, +0.032] | −0.195 [−0.275, −0.100] | conditional recovery −0.067 [−0.109, −0.024] |
| **B − A** | held-out | **−0.031 [−0.055, −0.008]** | −0.070 [−0.139, −0.009] | −0.076 [−0.126, −0.028] | — |
| B − A | test | −0.026 [−0.059, −0.003] | — | +0.211 [+0.022, +0.489] | — |

**Held-out false-outage failures (taxonomy B):** π₀ 161, A 138, B 113. It remains the largest failure category for A.

**Probe:**
- When the models answer, they mostly classify the error correctly. Real errors answered correctly: π₀ 24/24 parsed, A 25/25, B 58/62.
- Many replies are tool calls instead of a letter (held-out real errors: π₀ 76/100, A 75/100, B 38/100), so the probe stays descriptive.

## Replication: A vs B over 3 training seeds from one persisted π₀ (H100, 2026-11-19; pre-registered protocol in protocol.md)
- **Setup:** π₀ rebuilt once (STAGE0, 94 steps), checksummed and stored on network volume `9ghq6ztmfl`. Pairs re-verified under it (K = 4) and datasets rebuilt (408 steps per arm, 3 epochs). A: lr 1e-4; B: lr 5e-6 with reference log-probs precomputed. No selection; each adapter evaluated once on test (619 episodes) and held-out (1,582).
- **Source:** `pooled.json` (seed-pooled paired base-task cluster bootstrap, 2,000 draws), from `pod_backups/replication_slim_20261119.tgz` (sha256 8aae5439…).

### Held-out (primary)
| | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| A overall | 0.647 | 0.718 | 0.705 | 0.690 |
| B overall | 0.732 | 0.730 | 0.731 | 0.731 |
| π₀ overall | — | — | — | 0.718 |
| **B − A** | **+0.085** | **+0.012** | **+0.026** | **+0.041 [0.009, 0.075]** |

| pooled held-out | A | B | π₀ | B − A [95% CI] |
|---|---|---|---|---|
| fault-free (S0) | 0.626 | 0.690 | 0.632 | +0.064 [0.016, 0.114] |
| genuine repair | 0.780 | 0.890 | 0.860 | +0.110 [0.057, 0.168] |
| false-outage rate | 0.211 | 0.096 | 0.355 | −0.115 [−0.157, −0.067] |
| correct escalation | | | | −0.087 [−0.127, −0.048] |
| premature abstention | | | | −0.037 [−0.068, −0.006] |

- **Against π₀ on held-out:** A − π₀ = −0.028 [−0.048, −0.009]; B − π₀ = +0.013 [−0.016, 0.044].
- **Test:** A 0.876, B 0.866, π₀ 0.807. B − A = −0.010 [−0.051, 0.034]. Both arms beat π₀ (A − π₀ +0.069, B − π₀ +0.059; both CIs exclude 0).

### Decision rules (as pre-registered)
- **"A > B confirmed"** (pooled B − A CI < 0 and at least 2 of 3 seeds with B − A < 0): **not met.** The held-out difference has the opposite sign in all 3 seeds, and its pooled CI excludes 0 in B's favour. The seed-0 pilot finding (A > B, −0.031) **did not replicate**. The rules pre-registered only a test of A > B, so "B > A" is reported as an observed, unregistered direction.
- **Targeted false-outage intervention on A** (false-outage, taxonomy B, is the largest held-out failure category in at least 2 of 3 A seeds): **not met.** It is largest in 1 of 3 seeds.

  | A seed | largest held-out failure category (next) |
  |---|---|
  | s0 | G general 148 (D 143, B 127) |
  | s1 | B false outage 120 (D 118, C 112) |
  | s2 | C argument 123 (B 113, G 110) |

  B's held-out false-outage failures: 88 / 85 / 86.

### Reading
- B (verified DPO) is **stable across seeds** on held-out: overall 0.730–0.732. Relative to π₀ it cuts the false-outage rate by about 3.7×, but buys this partly with less escalation.
- A (balanced SFT at lr 1e-4) is **seed-sensitive**: held-out overall ranges 0.647–0.718. On held-out it ends **below π₀**, while beating it on test, which indicates overfitting to the training families. The original seed-0 A (0.763) came from a different π₀ rebuild, so π₀ variance is confounded with that earlier comparison.
- **Evaluation noise:** rerunning the same seed-0 adapters gave held-out overall A 0.640 vs 0.647 and B 0.731 vs 0.732. vLLM greedy decoding is not bit-deterministic, giving about 0.007 run-to-run variation, which is well below the B − A effect.

## Headroom over B (`scripts/analyze_b_headroom.py`, CPU, on the replication traces; descriptive, not pre-registered)
**Held-out (1,582 episodes):**
- **B is nearly deterministic across seeds:** 1,122 episodes are correct in 3/3 seeds, 417 fail in 3/3, and only 43 differ.
- **Most of B's stable failures are competence limits, not recovery failures.**
  - Of the 417 stable failures, π₀ also fails 357.
  - 382 fall outside S0. Of those, 327 (86%) are on tasks B cannot solve even fault-free (S0 0/3). By label: argument 111, general 90, ordering 62, false outage 52, missed escalation 12.
- **Recovery-specific failures** (B solves the same task fault-free in 3/3 seeds) number only **55**. They form one pattern, **over-persistence under outages**:
  - S2 false outage 18 and S2 general 9;
  - S3 missed escalation 10;
  - S2_act 8 and S2_obs 7.
- **B vs π₀:** 76 gains, mostly S1 false outage (20) and ordering (16), against 72 regressions, led by S3 missed escalation (18) and argument errors (27).
- **S3 behaviour:** π₀ escalates 338/342 and A 342/342; B escalates only 317/342. When it misses, B keeps re-issuing the failed call (6.7 vs 5.2 calls per S3 episode).
- **Ceiling:** fixing every recovery-specific B failure would raise held-out overall by only about **+0.03** (macro over 9 suites). The fault-free S0 rate (0.69) bounds everything else.

**Test (619):**
- 75 stable B failures. 37 are competence-limited and 34 recovery-specific, again S2/S2_obs persistence plus S3 missed escalation (9).
- Ceiling: about +0.04.

**Implication.** On this benchmark and with this model, the corrective-training question is close to saturated by B. The remaining recovery error is one behaviour (retrying through a persistent injected outage instead of switching endpoint or escalating), with at most about 0.03 held-out headroom. That is comparable to the pooled CI half-width, so even a perfect method would need many seeds to show a significant gain over B. The larger lever is fault-free competence on the held-out family, which is not a corrective-training effect.
