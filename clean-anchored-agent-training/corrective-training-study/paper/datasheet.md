# Datasheet: Uncertain-Outcome Faults (UOF) for tool-using agents
*Following the Datasheets for Datasets template (Gebru et al.). Release candidate, 2026-11-21.*

## Motivation
- **Purpose.** To measure whether a tool-using agent handles a state-changing call whose outcome is uncertain: a timeout or dropped connection after the request was sent.
- **Why existing benchmarks miss it.**
  - Standard fault suites (transient outages, persistent outages, impossible requests, argument and precondition errors) have unambiguous correct responses.
  - Uncertain commits are the case where **blind retry is harmful** (duplicates) and **giving up is harmful** (lost actions). Only check-then-decide is correct.
- **Created by:** the authors of the accompanying paper. No external funding is attached to the benchmark.

## Composition
**Base environments:**
- **BFCL v4 multi-turn base** (bfcl-eval 2026.3.23, Apache-2.0):
  - 398 usable single-turn episodes over 8 API classes;
  - executable backends;
  - state-equality grading.
- **τ²-bench airline** (sierra-research/tau2-bench @c06b07dd, MIT):
  - 50 tasks;
  - DB-hash grading;
  - used as a zero-shot second environment.

**Instances.** Each instance is an *outcome-twin pair* sharing an identical observable history:

| variant | what happened | correct behaviour |
|---|---|---|
| ACa | the faulted call **was applied** | check, do not repeat, finish |
| ACl | the faulted call **was not applied** | check, re-issue, finish |

| subset | count | criterion (by execution) |
|---|---|---|
| BFCL diagnostic | 116 tasks | executing the fault target twice breaks the end state, and a read-only function shows the difference |
| BFCL decisive | 73 tasks | an argument value of the call appears in one read-only output only if applied (outcome readable from one check) |
| τ² diagnostic | 13 write events / 8 tasks | non-idempotent and readable (11 `cancel_reservation`, 2 `book_reservation`) |
| τ² control | 36 write events | repeat-safe (checking unnecessary; measures over-checking) |

**Error wordings:**

| wording | text | used for |
|---|---|---|
| explicit | "…the operation may or may not have been applied." | train / eval |
| para | "…the request might already have been processed." | **eval only** |
| implicit | "Error: the request timed out." | **eval only**; the realistic case |
| train1, train2, t_gateway, t_dropped | — | **train only** |

**Controls shipped with every evaluation:**
- S0 (fault-free);
- S1 (transient outage, where retrying is correct);
- the original BFCL suites (S0–S3, real precondition and argument repair) on the test and held-out splits.

**Splits.**
- **BFCL train / dev / test / held-out** follow the parent project's pre-registered split. The held-out family is VehicleControlAPI for the standard suites.
- **For uncertain-outcome training studies,** splits are by **API class** (leave-one-class-out), fixed before training.

**Labels.**
- Correctness is computed by executing the agent's calls and comparing end states. No human labels.
- MCP-style tool annotations (`readOnlyHint`, `idempotentHint`) are derived by execution for 78 functions (`results/tool_annotations.json`).

**Sensitive content.** None. Tasks are synthetic API scenarios from BFCL and τ²-bench with fictional users. No personal data.

## Collection and validation
- **Construction is fully programmatic** (`scripts/feasibility_ambiguous_commit.py`, `scripts/tau2_airline_feasibility.py`).
- **Validation by scripted reference policies on every instance and wording:**

  | policy | ACa | ACl |
  |---|---|---|
  | verify-then-decide | 1.0 | 1.0 |
  | blind retry | 0.0 | 1.0 |
  | never retry | 1.0 | 0.0 |
  | escalate | 0.0 | 0.0 |

  - BFCL: 116/116 tasks for each policy.
  - τ²: 39/39 events per twin.
  - τ² control: blind retry passes both twins.
- **Unit tests** cover twin structure, decisiveness, leakage of evaluation tasks, classes and wordings into training data, and determinism (`tests/`).

## Preprocessing
- **BFCL:** one user turn per episode; earlier turns shown as executed history; fault target = the first state-changing call of the turn.
- **τ²:** forced prefix of gold actions before the faulted write. The customer request is rewritten to first person from the task's scenario. The official LLM user simulator is not used (a limitation; see the paper).

## Uses
- **Intended:** evaluating and training agents' recovery from uncertain-outcome failures; measuring over-checking cost; transfer studies across API classes and environments.
- **Not intended:**
  - as a measure of general tool-use competence (S0 is reported separately for that reason);
  - as evidence of safety on real payment or booking systems.

## Distribution and licence
- **Released with:** code, graders, training-data builders, all pre-registered protocols, per-episode evaluation traces and summary archives (sha256-listed).
- **Licences:**
  - BFCL-derived content keeps BFCL's Apache-2.0;
  - the vendored τ²-bench code and data keep MIT (LICENSE included, unmodified);
  - new code and data: Apache-2.0 (proposed).
- **Not redistributed:** model weights larger than the LoRA adapters. π₀ and π_B are reproducible from the released scripts.

## Maintenance
- **Pinning:** versions are pinned by commit and sha256 (bfcl-eval 2026.3.23; tau2-bench c06b07dd).
- **Changes:** new wordings, environments or tasks are released as new versioned subsets. Earlier versions stay frozen so results remain comparable.
- **Known limitations:**
  - 8 BFCL API classes;
  - small held-out-class sets (9–35 tasks after conditioning on fault-free success);
  - a small, simulator-free τ² subset;
  - faults are injected at the first state-changing call only.
