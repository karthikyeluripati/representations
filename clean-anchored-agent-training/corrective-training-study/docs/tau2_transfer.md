# Zero-shot transfer to τ²-bench airline. Pre-registered 2026-11-21, before any model is evaluated on it
**Purpose:** a second environment (different codebase, task source and API style) for the uncertain-outcome results. **No τ² data is used in any training.** The models are frozen adapters from the BFCL / synthetic studies.

## Environment (`src/environment/tau2_airline.py`; vendored tau2-bench @c06b07dd, MIT)
- **Events** (`results/tau2/airline_write_events.json`, by execution):
  - **13 diagnostic** (non-idempotent, readable) writes in 8 tasks: 11 `cancel_reservation` (repeating refunds again), 2 `book_reservation`;
  - **36 control** writes that are safe to repeat (updates, guarded bookings).
- **Episode:**
  - system = the std recovery prompt + the airline policy;
  - user = the task's customer request (reason, known info, instructions; first person);
  - history = the gold actions before the write, as the agent's own calls;
  - the write returns an uncertain-outcome error (ACa applied / ACl lost); S0 = no fault.
- **Grading:** τ²'s DB hash vs the gold end state, and no escalation or transfer to a human.
- **Scripted validation** (`scripts/tau2_validate.py`):

  | policy | diagnostic ACa | diagnostic ACl | control ACa / ACl |
  |---|---|---|---|
  | verify-then-decide | 39/39 | 39/39 | 108/108 each |
  | blind retry | 0/39 | 39/39 | 108/108 each (harmless) |
  | never retry | 39/39 | 0/39 | ACl 0/108 |

## Policies (frozen)
- π_B;
- SYNOLD s0–s2 (control);
- SYNIAR s0–s2 (32 synthetic classes);
- SYN64IAR s0–s2 (64 classes).

## Metrics and rules (fixed now)
- **Primary:** on diagnostic events, **implicit** wording, the rate of checking right after the error, and AC success given S0. Pooled over 3 seeds per arm, compared with SYNOLD by an event-cluster bootstrap.
- **Transfer observed** if the pooled SYNIAR (and/or SYN64IAR) check rate after the implicit error exceeds SYNOLD's with a CI excluding 0.
- **Success is reported descriptively.** With 13 events the success CIs will be wide.
- **Secondary:**
  - para and explicit wordings;
  - S0;
  - control events: success and extra calls (over-checking cost).
- **Known limitations:**
  - no LLM user simulator: a fixed request is given and the prefix is forced;
  - 13 diagnostic events;
  - the airline policy text is long (prompt shift vs training).

## Result (2026-11-21; `pod_backups/tau2_eval_slim_20261121.tgz`, sha256 01ffa196…)
**Primary** (diagnostic events, implicit wording; rate of checking right after the error, pooled over 3 seeds):

| policy | check rate | per seed |
|---|---|---|
| π_B | 0.000 | — |
| SYNOLD | 0.000 | 0 / 0 / 0 |
| SYNIAR (32 classes) | 0.000 | 0 / 0 / 0 |
| SYN64IAR | 0.026 | 0 / 0.077 / 0 |

- **SYN64 − SYNOLD = +0.026 [0.000, 0.077]. Transfer was not observed** (the CI does not exclude 0).
- **Para / explicit:** checks 0.05–0.08 for the synthetic arms vs 0–0.03 for the control.
- **S0 = 0.15** for every policy (2 / 13 events). Without a user simulator the 4B agent rarely completes the remaining airline actions, so success is uninformative; the check rate is the measure.
- **No episode hit the context limit.**

**Diagnosis, checks by airline function (all wordings, pooled):**

| function | SYNIAR | SYN64IAR | SYNOLD | π_B |
|---|---|---|---|---|
| `book_reservation` (create-type) | 10 / 36 | 10 / 36 | 2 / 36 | 0 / 12 |
| `cancel_reservation` (non-idempotent, refunds again) | **0 / 198** | **0 / 198** | 0 / 198 | 0 / 66 |

**Reading:**
- The learned rule is keyed to **verb semantics**, not to side effects. In every synthetic API, *create*-type calls were non-idempotent and *cancel / remove / delete* calls were idempotent.
- The model therefore checks before re-booking (10/36), but never before re-cancelling, even though τ²'s `cancel_reservation` refunds again on every call.
- This is a concrete, testable limitation of the synthetic curriculum. **Next:** synthetic APIs whose verb ↔ idempotency mapping is varied (non-idempotent cancels with refunds, idempotent creates with client keys), so that only the tool's documented or observed side effects predict the right action.
