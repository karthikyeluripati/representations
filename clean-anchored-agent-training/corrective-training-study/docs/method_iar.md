# Method study 3: Idempotency-Aware Recovery (IAR). Pre-registered 2026-11-20, before any IAR training
**Motivation** (method_egr.md):
- Plain SFT on 4 diverse API classes transfers *checking and reading* to an unseen class: 76/76 checks and 64/76 correct after the check on the unseen paraphrase.
- **No policy ever checks after the realistic implicit timeout** "Error: the request timed out.": 0/76 and 0/40.
- Hypothesis: the training convention "timeout-type outage, so retry" (S1 anchors) teaches that timeouts mean *not applied*. Physically, a timeout on a state-changing call leaves the outcome unknown.

## The rule taught by IAR

| failure on … | error type | IAR target | old convention (baseline) |
|---|---|---|---|
| a **state-changing** call (non-idempotent) | timeout / dropped connection | **check, then decide** (outcome-twin decisions) | retry directly |
| a state-changing call | clean rejection ("503 Service Unavailable") | retry directly | retry directly |
| a **repeat-safe** call (idempotent) | any | retry directly | retry directly |
| any call, explicit "may or may not have been applied" wordings | — | check, then decide | check, then decide |

## Data (fold T, `scripts/build_iar_data.py`; execution-verified; train split of MessageAPI, TradingBot, TwitterAPI, TicketAPI only)
- **IAR (517):**
  - 198 twin decisions on explicit/train1/train2;
  - 132 twin decisions on the timeout-type TRAIN strings;
  - 22 rejection retries;
  - 165 repeat-safe retries.
- **Baseline AAC3 (517, count-matched by cycling):** 198 twin decisions + 66 direct write retries after all three TRAIN strings + the same 165 repeat-safe retries.
- **The sets differ only in how a timeout-type failure of a state-changing call is answered** (unit-tested).
- **Both:** + the same 84 B-anchor decisions; SFT from π_B; LoRA r 8; lr 1e-4; 3 epochs; batch 4; seed 0.
- **The evaluation strings are never trained:**
  - implicit "Error: the request timed out.";
  - para;
  - the V2 S1 pool.

## Outcomes and rules (fixed now)
- **Primary:** success on ACa/ACl with the **implicit** wording, std prompt, on held-out TravelAPI decisive tasks that π_B solves fault-free (35 tasks). Compared as IAR − AAC3, task bootstrap.
- **Seed-0 gate for seeds 1–2.** All must hold:
  - IAR − AAC3 ≥ **+0.15** on the primary;
  - para (held-out Travel) not lower than AAC3 by more than 0.05;
  - S1|S0 ≥ 0.95.
- **Reported:**
  - extra calls on S1/S2 (the cost of checking, since IAR may check after timeouts on writes in S1, which is correct but costs one call);
  - S0;
  - original test + held-out overall vs B_s0 (0.872 / 0.732);
  - checks after the implicit error.
- **A later claim (3 seeds) needs:**
  - the pooled primary CI > 0;
  - at least 2 of 3 seeds positive;
  - original overall within 0.02 of B_s0;
  - S1|S0 ≥ 0.95.

**Budget:** 2 SFT runs (about 25 min each), evaluations and setup. About 1.5 H100-h, about $7.

## Seed-0 result (2026-11-20; `pod_backups/iar_s0_slim_20261120.tgz`, sha256 c30648f4…)
**Primary** (implicit "Error: the request timed out.", held-out TravelAPI, 35 tasks):
- IAR 0.486, AAC3 0.486, π_B 0.514. **IAR − AAC3 = 0.000.**
- **Gate failed.** Para also dropped: IAR 0.643 vs AAC3 0.857, −0.214 [−0.314, −0.100]. S1|S0 0.98 (ok).
- **Not continued.**

**Checks after the implicit timeout, by class** (decisive tasks; "trained task" = a train-split task of a training class):

| class | trained tasks: baseline | trained tasks: IAR | untrained tasks: baseline | untrained tasks: IAR |
|---|---|---|---|---|
| MessageAPI | 2/24 | **24/24** | 0/16 | **14/16** |
| TicketAPI / TradingBot / TwitterAPI | 4/20 | **20/20** | 1/10 | **10/10** |
| **TravelAPI (held-out class)** | — | — | 0/76 | **0/76** |

**Finding:**
- IAR teaches the model to treat a plain timeout as an uncertain outcome. The skill **transfers to new tasks and to an unseen wording within the trained API classes** (from about 0% to 96%).
- It **does not transfer to an unseen API class at all** (0/76).
- So the learned rule is keyed to the **identity of the API functions** ("timeouts on send_message are risky"), not to the semantic property (the call is state-changing and non-idempotent). Nothing in an unseen class's tools tells the model which calls are non-idempotent.
- By contrast, an **explicit uncertainty cue in the error message transfers across classes**: para gives 76/76 checks on Travel for AAC2/AAC3.
- **Original benchmark** (IAR, AAC3): held-out 0.740 / 0.733 and test 0.834 / 0.864, against B_s0 0.732 / 0.872.
