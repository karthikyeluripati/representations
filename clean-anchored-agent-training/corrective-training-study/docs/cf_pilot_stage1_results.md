# CF-SFT pilot, stage 1: seed-0 training and futility gate (2026-11-22)

**Protocol:** `cf_sft_pilot_protocol.md`, frozen at `657f44a`. The protocol, the gates, the data and the eval code were not changed. The gate was applied mechanically by `scripts/cf_pilot_analysis.py --stage futility`.

**Evidence:**
- **Archive:** `pod_backups/cf_pilot_stage1_20261122.tgz`, sha256 `d1ccf5834cc939602828f288077f82c0291f8f74e219730f412e25c264ae520c`. The checksum is identical on the pod and locally. The archive holds the traces, the logs, the render manifests, the run configs, the log histories and the driver.
- **Adapters:** kept on network volume `9ghq6ztmfl`:
  - CF_s0 `adapter_model.safetensors` sha256 `6cfecfbb…b3b56b3`;
  - CTRL_s0 `dfa15a5e…5bdac5a3`.
- **Outputs:** `data/cf_pilot_stage1/futility.json` (the gate) and `data/cf_pilot_stage1/descriptive.json` (paired differences, held-out cells, the raw-row fault-free check). Both were recomputed locally from the archived raw traces.

**Run facts:**
- **Pod:** one H100 SXM, both arms run one after the other.
- **CF_s0 training:** 3,787 s, 1,077 steps, recipe identical to CAW-v2.
- **Pre-flight checks:** π_B checksum, the tests, and the spec and data hashes all passed on the pod before training.
- **Cost:** about **$12.9** (cap $14).
  - The cost guard stopped the pod at 170 min. The driver had in fact already finished (`STAGE1_ALL_DONE`, with all traces written), but the guard did not detect it.
  - The pod was then restarted for about 5 min, only to archive the outputs. Nothing was re-run.

## 1. Futility gate (pre-registered, seed 0)
| check | criterion | value | result |
|---|---|---|---|
| CF E2 | ≥ 0.40 | 0.481 [0.463, 0.500] | pass |
| CF − CTRL on E2 | ≥ +0.20 | +0.481 [0.463, 0.500] | pass |
| CF fault-free | ≥ 0.95 | 0.996 [0.989, 1.000] | pass |
| CF feasible final state | ≥ CTRL − 0.05 | 0.961 vs 0.951 | pass |

**Mechanical decision: CONTINUE TO STAGE 2.**

## 2. T2-test (20 held-out families, 4,320 episodes per arm; 95% family-cluster CI, paired)
| metric | CF_s0 | CTRL_s0 | CF − CTRL |
|---|---|---|---|
| **effect sensitivity (E2)** | **0.481** [0.463, 0.500] | 0.000 | **+0.481** [0.463, 0.500] |
| **held-out-combination twins (E1)** | **0.722** [0.709, 0.734] | 0.500 | **+0.222** [0.209, 0.234] |
| outcome-twin correct (all cells) | 0.833 [0.825, 0.840] | 0.778 | +0.055 [0.047, 0.062] |
| **safe escalation (infeasible)** | **0.125** | 0.000 | +0.125 |
| reconciliation twins | 0.944 [0.919, 0.969] | 0.000 | +0.944 |
| keyed retry correct | 1.000 | 1.000 | 0 |
| final state, feasible | 0.961 [0.955, 0.966] | 0.951 | +0.010 [0.004, 0.015] |
| fault-free success | 0.996 [0.989, 1.000] | 1.000 | −0.004 [−0.011, 0.000] |
| premature escalation | 0.024 | 0.000 | +0.024 |
| unnecessary checks | 0.021 | 0.641 | −0.619 |
| excess calls per episode | +0.062 | +0.483 | −0.420 |
| decision accuracy | 0.887 | 0.867 | +0.020 |

For reference, the stage-0 result for every baseline (π_B, SYN64, CAW, CAW-v2, all prompts) was E2 = 0, E1 ≈ 0.50, safe escalation 0 and reconciliation 0.

**Accuracy by oracle action (CF / CTRL):**

| oracle action | CF | CTRL |
|---|---|---|
| do_once | 0.996 | 1.000 |
| reissue | 0.965 | 1.000 |
| reissue_same_key | 1.000 | 1.000 |
| check_then_branch | 0.809 | 1.000 |
| reconcile | 0.972 | 0.500 |
| **escalate** | **0.125** | 0.000 |

**Held-out combinations, timeout cells (CF / CTRL; 160 episodes each):**
- (keyed, none, absent): 1.00 / 1.00. Not diagnostic, because blind same-key retry is also correct here.
- (keyed, none, present): 1.00 / 1.00. Not diagnostic, for the same reason.
- (nonidem, none, present): **0.944 / 0.50**. This cell is the entire E1 gain.
- (nonidem, stale, absent): **0.00 / 0.00**. The oracle action is escalate.

**Where CF fails: escalation.**
- In the 320 escalate-oracle episodes, CF re-issued the non-repeat-safe write in 280 and escalated in 40.
- All 40 escalations are in the trained cell (nonidem, none, absent). There it reaches 0.25.
- CF never escalates in the held-out stale-verifier cell.
- So CF learned "reconcile when compensation exists" and "do not over-check". It did not learn "stop when the outcome cannot be resolved".

## 3. Frozen T1 cross-template test (unchanged; 20 families, 1,920 episodes)
| metric | CF_s0 | CTRL_s0 | CAW-v2 (2 seeds) | CF − CAW-v2 |
|---|---|---|---|---|
| **effect sensitivity (E3 / M1)** | **0.087** [0.044, 0.138] | 0.000 | 0.000 | +0.087 [0.044, 0.138] |
| safe escalation (M3) | 0.119 [0.069, 0.163] | 0.000 | 0.003 | +0.116 [0.069, 0.159] |
| history twins (M2) | 0.988 | 0.963 | 0.944 | +0.044 [−0.013, 0.106] |
| final state, feasible | 0.989 | 0.996 | 0.993 | −0.004 [−0.014, 0.007] |
| fault-free success | 1.000 | 1.000 | — | — |
| unnecessary checks | 0.217 | 0.600 | 0.548 | −0.331 |

**Cross-template transfer is weak.** The final gate S3 needs E3 ≥ 0.50. At seed 0, CF's upper confidence bound is 0.138.

## 4. Independent fault-free check (raw trace rows, `error == none`; not via `metrics()`)
| run | episodes | goal state reached | no duplicate | no escalation | mean calls (min 1.0) |
|---|---|---|---|---|---|
| T2 CF_s0 | 720 | 720 | 720 | 717 | 1.017 |
| T2 CTRL_s0 | 720 | 720 | 720 | 720 | 1.000 |
| T1 CF_s0 | 320 | 320 | 320 | 320 | 1.066 |
| T1 CTRL_s0 | 320 | 320 | 320 | 320 | 1.000 |

This agrees with the metric module. On fault-free episodes, CF's only failures are 3 unnecessary escalations on T2 out of 720. It also makes a few extra read calls on T1.

## 5. Reading, stated against the pre-registered criteria
- **The gate passed.** Effect sensitivity rose from 0 for every existing recipe and prompt to 0.48 on held-out families. The gain over the matched control is large and well outside the CIs.
- **The control did what it was designed to do.** It reproduces the CAW rule exactly (initial decision accuracy 1.000, E2 = 0). The gain therefore comes from the counterfactual labels, not from more training or extra data.
- **Seed 0 has three weaknesses that bear on the stage-2 final gates.** These are observations, not gate results:
  1. **Safe escalation is close to absent** (0.125 T2, 0.119 T1). CF re-issues non-repeat-safe writes in 88% of unresolvable episodes.
  2. **E1 = +0.222 rests on one held-out cell.** Two of the four held-out combinations are non-diagnostic, and the stale-verifier combination is 0 for both arms. S2 (≥ +0.20, every-seed ordering) is borderline.
  3. **Cross-template E3 = 0.087**, far below S3's 0.50. If S1, S2 and S5 then pass, the pre-registered map gives **CROSS-TEMPLATE RESIDUAL**.
- **Not yet known:** seed variance; whether the gain comes from reading the contract text (S5 needs ABL); regression on BFCL and τ² (S4). Stage 2 measures these.
- **Limits:**
  - one seed;
  - one model family;
  - synthetic templates;
  - all T2 families share one template, so T1 is the only cross-template check.
