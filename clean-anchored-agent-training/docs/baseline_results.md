# Baseline results

**No model has been trained or evaluated in this project yet.** No GPU was authorized, and the previous project's weights are gone. Everything below is CPU validation of the infrastructure. It is not policy evidence.

## Grader and metric validation: scripted policies through the model evaluation loop
| policy | split | n episodes | fault-free | real-error repair (RP+RA) | injected recovery | correct escalation (S3) | premature abstention | false-outage rate |
|---|---|---|---|---|---|---|---|---|
| oracle | held-out | 1,582 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 (100 events) |
| oracle | test | 619 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 (60) |
| outage-rule surrogate | held-out | 1,582 | 1.000 | **0.000** | 1.000 | 1.000 | 0.081 | **1.000** (304) |
| outage-rule surrogate | test | 619 | 1.000 | **0.000** | 1.000 | 1.000 | 0.122 | **1.000** (276) |

The surrogate behaves the way the previous project's recovery-SFT policy was measured to behave on 38% of real errors: it answers real API errors with retry, backup or escalate. It is perfect on injected outages and fails every real-error repair, and the metrics separate this exactly.

## Data available for the pilot (`data/cacrt_pairs_v1/MANIFEST.json`)
| split | corrective states (RP / RA) | states usable as pairs or eval episodes | base tasks with a state |
|---|---|---|---|
| train | 50 / 260 | 39 / 223 → **709 verified pairs** (retry 262, escalate 262, alternative 185) | 72 of 89 |
| dev (curves only) | 15 / 84 | 14 / 69 | 25 of 29 |
| test | 9 / 67 | 7 / 53 | 15 of 22 |
| held-out (Vehicle) | 81 / 27 | 81 / 19 | 44 of 49 |

**Twin anchors:**
- **Train:** 262 T-out + 262 T-clean, each verified by execution.
- **Generic anchors:** 492.
- **Dev twins:** 83 + 83.

**Continuation-dependence of the negatives** (the reason pairs must be re-verified on GPU with the policy itself):
- Under the competent repairer continuation, *retry* is never outcome-critical for RA: 0 of 260 fail.
- *Escalation* always is critical: 260 of 260 fail.
- So the CPU pairs encode "retry is bad" only relative to an outage-procedure continuation.

## Arms implemented and smoke-tested on a tiny random Qwen3 (CPU, 2 steps each)
- **Arms:** STAGE0, B1, B2, B3, B4 (PAFT-style, programmatic), EXP1, B5, EXP2 (= B6, CSO-style), B7_kl, B7_replay, B8_pcgrad, EXP3 (CACRT), EXP3_agem, EXP3_unverified.
- **B0** is the frozen base and needs no training.
- **Not implemented:** a Fission-GRPO baseline. It needs on-policy RL with a 32B feedback simulator; the simulator weights are unreleased and the default feedback is gpt-4.1, a paid API. A scaled-down reimplementation is listed in `paper_plan.md` as a qualified comparison.
