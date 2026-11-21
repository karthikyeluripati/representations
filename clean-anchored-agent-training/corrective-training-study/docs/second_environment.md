# Second environment: τ²-bench (feasibility, CPU, 2026-11-20)
**Source:** sierra-research/tau2-bench @c06b07dd (MIT, last commit 2026-11-10).
- **Domains read:** retail (114 tasks; official split train 74 / test 40) and airline (50).
- **Grading:** `reward_basis`:
  - retail: DB + NL assertion;
  - airline: DB + communicate.

  The DB check replays the gold actions and compares database states, which is the same principle as BFCL's state equality.

## Ambiguous-commit fit, by write tool (from the tool code)
| domain | tool | second call after success | AC-diagnostic? |
|---|---|---|---|
| airline | `book_reservation` | new reservation id + new payment | **yes** (10 gold uses) |
| airline | `cancel_reservation` | status set again, refunds appended again (no "already cancelled" guard visible) | **likely**; must be confirmed by execution |
| airline | `send_certificate` | adds another certificate | **yes** |
| airline | `update_reservation_flights` / `_baggages` | charge computed from the difference with the current state, so 0 | probably idempotent (control) |
| retail | `cancel_pending_order`, `exchange_…`, `return_…`, `modify_pending_order_items` | raises (status no longer pending or delivered); DB unchanged | **no**: guarded APIs, a natural control where verification is unnecessary |

## What this gives the paper
- **Airline:** about 20 tasks where blind retry corrupts the DB. This is in a different codebase, task source and dialogue setting from BFCL, so it satisfies bar item 4 ("second environment").
- **Retail:** a **guarded-API control**. A good method should not add needless verification calls where the API already prevents duplicates. This measures over-verification cost.

## Costs and risks
- **The official protocol uses an LLM user simulator.** The plan is to serve one locally (vLLM OpenAI-compatible server, a fixed open model, for example Qwen3-30B-A3B-Instruct-2507). Simulator choice is a confound, so it is fixed before any result and the same simulator is used for every agent.
- **Engineering:**
  - the fault-injection wrapper around the tool executor;
  - our agent loop speaking τ²'s tool schema;
  - an execution check of each "likely" row above.
- **Estimate:** about 1–2 days of CPU work, then a small GPU run.
- **Small task counts** (about 20 diagnostic airline tasks). This needs task augmentation, e.g. several fault positions per task or initial-state variants, defined before training.
- **Not yet examined:** AppWorld, ToolSandbox.

## Feasibility by execution (2026-11-21; `scripts/tau2_airline_feasibility.py`, `results/tau2/airline_write_events.json`)
- **Setup:** τ²-bench vendored at c06b07dd (`third_party/tau2_c06b07dd`). The airline environment runs locally (14 tools; gold actions replay and change the DB hash).
- **Method:** for every gold write action, execute it once vs twice from the same state, and test whether a read the agent can call (`get_user_details` / `get_reservation_details`, ids known from context) shows the change.

| write | repeating after success | readable | events |
|---|---|---|---|
| `cancel_reservation` | **non-idempotent** (refunds appended again) | yes | 11 |
| `book_reservation` | non-idempotent 2; in 8 the second booking errors and the DB is unchanged | yes | 10 |
| `update_reservation_flights / _baggages / _passengers` | idempotent | yes | 28 |

- **Diagnostic set:** 13 non-idempotent, readable write events in 8 tasks.
- **Control:** 28 idempotent events, where checking is unnecessary (over-checking cost).
- **Use in the paper: a zero-shot transfer test.** Models trained only on BFCL + synthetic APIs are evaluated on τ² airline uncertain-outcome faults without any τ² training.
- **Open design choice, to be pre-registered before running:**
  - the official protocol uses an LLM user simulator; a fault-injection episode (prefix of gold actions executed, then the fault on the write) avoids it;
  - the user request is given from the task's instructions.
