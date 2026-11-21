# Did It Go Through? Tool Agents Blindly Retry Uncertain Writes, and What Training Can and Cannot Fix

*Working draft, target NeurIPS 2027 Datasets & Benchmarks track. Numbers are from the pre-registered study documents in `docs/`. Single-seed results are marked.*

## Abstract
- **Problem.** When a state-changing tool call fails with an uncertain outcome (a timeout after the request was sent), retrying blindly can duplicate a booking, an order or a payment, and giving up can lose it. The safe policy is to *check the state, then decide*.
- **Benchmark.** We introduce **uncertain-outcome faults** on BFCL v4 multi-turn tasks, graded by state equality. Each fault comes as a pair of *outcome twins* with an identical observable history: the call was applied (ACa) or lost (ACl). Only check-then-decide solves both.
- **Untrained models.** Across Qwen3 models from 4B to 30B-A3B, with and without an explicit instruction, agents re-issue the call in 113–115 of 116 cases and score at the blind-retry level (0.47–0.53, where 1.0 is achievable). Corrective training on standard faults, including a CSO-style verified-DPO baseline, does not change this.
- **Five pre-registered training interventions** give a consistent picture of what transfers to unseen API classes:
  - (i) *checking* transfers when the error message states the uncertainty;
  - (ii) *reading* the check transfers only with training on diverse API classes;
  - (iii) *recognising* an implicit risk (a plain "request timed out") stays tied to API identity: about 96% within trained classes, 0% on a held-out class, even when MCP-style idempotency annotations are printed in every tool description;
  - (iv) rationale supervision and DPO both hurt, while SFT does not;
  - (v) **training on 32 synthetic stateful APIs** that vary surface form but keep the create / set / remove semantics fixed is the first intervention that makes the implicit rule transfer: +0.10 [0.02, 0.17] over 3 seeds, +0.17 on an unseen paraphrase, with no regression.
- **Release.** The benchmark, graders, training data builders and every protocol.

## 1. Introduction
- **Tool-using agents act on real systems.** Real systems fail ambiguously: a request can time out *after* the server applied it. Distributed-systems practice handles this with idempotency keys and read-after-write checks. Language-model agents are usually trained and evaluated on faults whose correct response is unambiguous: retry a transient outage, switch to a backup, repair a wrong argument, or escalate.
- **Our motivating finding is negative.** On BFCL multi-turn tasks with injected outages, a strong corrective-training pipeline (π₀ SFT, then CSO-style step-verified DPO) leaves only about +0.03 of recoverable held-out headroom. Most remaining failures are competence-limited: 86% fail even without faults. The standard fault suite is saturated.
- **A fault class where retrying is harmful.** The ambiguous commit is the natural case. No model we tested handles it, whatever the size or prompt, and blind retry happens almost every time.
- **What we ask and do.** Can training fix it in a way that *generalises* to APIs the agent was not trained on? We answer with four pre-registered interventions, each with a single-seed futility gate and a matched baseline. We report all four, including the failures, because together they locate where generalisation breaks.

**Contributions.**
1. **Uncertain-outcome faults** (§2): outcome-twin construction, a decisive subset where the outcome is readable from one check, wording shifts (explicit, unseen paraphrase, implicit), controls where retrying is correct, and execution-verified graders (scripted reference policies separate exactly).
2. **A capability gap** that neither prompting nor scale closes (§4).
3. **Transfer analysis of training** (§5), separating checking, reading and risk recognition across held-out API classes.
4. **Methodology:** pre-registered gates, matched arms, execution-verified data. These catch several design errors before they reach a result (§6).

## 2. Benchmark
- **Environment.** BFCL v4 multi-turn base tasks (bfcl-eval 2026.3.23, executable backends). One user turn per episode, with earlier turns shown as executed history.
- **Grading.** State-based: all public attributes of all involved API instances must equal the ground-truth end state.
- **Scope.** 398 usable tasks over 8 API classes. Held-out family for the standard suites: VehicleControlAPI.

**Fault.** The first call to the task's fault target *f* (its first state-changing call) returns an error. In **ACa** the call was executed; in **ACl** it was not. The observation is identical in both.

| wording | text | role |
|---|---|---|
| explicit | "…request timed out after the server received it; the operation may or may not have been applied." | train / eval |
| para | "…connection lost before a response arrived; the request might already have been processed." | eval only (unseen) |
| implicit | "Error: the request timed out." | eval only (unseen; realistic) |

**Task selection** (CPU, by execution):
- **Diagnostic:** executing *f* twice breaks the end state, and some read-only function shows the difference. 116 / 398 tasks, e.g. book_flight, send_message, place_order, purchase_insurance.
- **Decisive:** some argument value of *f* appears in a read-only output after *f* and not before, so the outcome can be read from one check without knowing the prior state. 73 tasks.
- **Excluded:** `place_order`, whose order history lists only ids.

**Validation.** Scripted policies on all 116 tasks:

| policy | ACa | ACl | S0 |
|---|---|---|---|
| verify-then-decide | 116 / 116 | 116 / 116 | 116 / 116 |
| blind retry | 0 | 116 | 116 |
| never retry | 116 | 0 | 116 |
| escalate | 0 | 0 | 116 |

The same separation holds on every wording.

**Metric.** Mean success over ACa and ACl, on tasks the policy (or a fixed reference) solves fault-free, to separate recovery from competence. Blind retry = 0.50; verify-then-decide = 1.00.

**Controls:**
- S0 (fault-free);
- S1 (transient outage, where retrying is correct);
- the original suites S0–S3, RP/RA on test and held-out.

## 3. Policies and training setup
- **Model and π₀.** Qwen3-4B-Instruct-2507 (π₀ = base + LoRA r32 SFT on fault-free and outage trajectories).
- **Corrective baselines on the standard faults, reproduced over 3 seeds:**
  - A: balanced corrective SFT;
  - B: CSO-style verified DPO (step-level pairs with K=4 rollout verification).
- **Larger models (no training):** Qwen3-8B (thinking off) and Qwen3-30B-A3B-Instruct-2507 (FP8).
- **Training.** All method studies start from π_B (π₀ + B merged) and use TRL 0.24 with LoRA r8.
- **Data.** Every training state and action is **verified by execution**: the chosen continuation succeeds, and the rejected one fails in the relevant twin.

## 4. The gap: prompting and scale do not help
On 116 diagnostic tasks, AC success given S0:

| policy | prompt | explicit | para | implicit | S1\|S0 |
|---|---|---|---|---|---|
| base 4B | std / +check | 0.49 / 0.50 | 0.50 / 0.51 | 0.50 / 0.50 | 0.97 / 0.98 |
| π₀ | std / +check | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 1.00 |
| B (verified DPO) | std / +check | 0.51 / 0.51 | 0.51 / 0.51 | 0.51 / 0.51 | 1.00 / 0.98 |
| Qwen3-8B | std / +check | 0.47 / 0.49 | 0.49 / 0.49 | 0.48 / 0.49 | 0.94 |
| Qwen3-30B-A3B | std / +check | 0.50 / 0.53 | 0.50 / **0.65** | 0.50 / 0.50 | 1.00 |

- **Blind retry dominates.** It is the next action in 113–115 of 116 ACa episodes for every 4B policy.
- **The instruction helps only the 30B model, and only partially.** Even when it checks (44 / 116 on para), it often re-issues *f* after reading a state that already shows it.

**Context on the standard faults** (3-seed replication; held-out):
- B (0.731) > A (0.690): +0.041 [0.009, 0.075], positive in 3 / 3 seeds. This reverses a single-seed pilot (−0.031).
- B's remaining recoverable headroom is about +0.03.

## 5. What training transfers
All rows below are seed 0, tested on held-out API classes and unseen wordings, against matched baselines.

| study | intervention | key result | gate |
|---|---|---|---|
| OTV | twin DPO (check at error; decide after check), 2 training classes | unseen classes 0.51 vs SFT 0.53; *checks* rise 0 → 58% but post-check decisions are at chance (28 / 67); in-dist 0.72 | failed |
| EGR | SFT with evidence rationales, 2-fold leave-one-class-out | −0.091 [−0.148, −0.034] vs plain SFT; over-checking (+1–1.9 calls on S0) | failed |
| plain SFT, 4 classes (EGR baseline) | | held-out Travel, para: checks 76 / 76, correct after check 64 / 76 | — |
| IAR | treat timeouts on writes as uncertain (rejections and repeat-safe calls: retry) | implicit timeout: checks 96% on trained classes (new tasks), **0 / 76** on held-out class | failed |
| IAR + MCP annotations | `readOnlyHint` / `idempotentHint` derived by execution, in every tool description | held-out implicit: 2 / 76 checks; in-dist 0.92; annotations alone: no effect | failed |
| **SYN-64** (64 classes, same data size) | as SYN with twice the class diversity | **held-out implicit 0.667 vs SYN-32 0.610: +0.058 [0.014, 0.110]**; vs control +0.156; checks 55 / 64 / 8 of 76 | **scaling supported** |
| **τ² airline (zero-shot)** | frozen SYN models on 13 non-idempotent airline writes | implicit checks: SYN-64 0.026 [0, 0.077], others 0; `book_reservation` 10 / 36 checks vs `cancel_reservation` **0 / 198** | **no transfer** |
| **SYN** (synthetic-API diversity) | IAR data + 32 synthetic stateful APIs (create = check, set/remove = retry), vs identical data under the old convention | **3 seeds: held-out implicit 0.610 vs 0.510, +0.099 [0.024, 0.171], positive in 3 / 3 seeds; checks 46 / 50 / 8 of 76 vs 0 / 0 / 0; para +0.171 [0.095, 0.238]**; original suites within 0.01 of B | **claim rule met** |

**Findings:**
1. **Checking is learnable and transfers across API classes when the error text states the uncertainty.**
2. **Reading the check needs diversity.** With 2 training classes, post-check decisions on unseen classes are at chance. With 4 classes, 84% are correct on a held-out class.
3. **Implicit-risk recognition is keyed to API identity.** "Timeouts on `send_message` are risky" is learned; "timeouts on non-idempotent calls are risky" is not, even when idempotency is printed beside each tool.
4. **Diversity, not annotation, unlocks the abstract rule, partially.** Adding 32 synthetic stateful APIs that vary surface form while keeping the create / set / remove semantics fixed is the only intervention that makes implicit-risk recognition transfer to an unseen real API class. Over 3 seeds: +0.099 [0.024, 0.171], checks 46 / 50 / 8 of 76 vs 0 in every other condition. The unseen paraphrase improves by +0.171, with no regression on the original suites. The effect is seed-variable, so the rule is learned but not yet reliably.
5. **Diversity scales, but the learned rule is verb-semantic.** At fixed data size, 64 classes beat 32 (+0.058 [0.014, 0.110]). Zero-shot on τ²-bench airline, the same models check before re-booking but never before re-cancelling (0 / 198), because cancel-type calls were idempotent in every synthetic API, while τ²'s cancellation refunds again. The rule follows verb meaning, not side effects. One training seed fails in every configuration, so reliability is a separate, open problem.
6. **Supervision form matters.** Rationale supervision causes over-checking and worse decisions. DPO variants regress ordinary behaviour (original test 0.74–0.80 vs 0.87). SFT stays within about 0.01 of B on the original suites in the best fold.

## 6. Methodology notes
- **Pre-registration.** Every study fixed its primary metric, matched baseline and futility gate before training. Each gate was applied as written.
- **Execution verification** caught design errors that would otherwise have produced misleading data:
  - non-decisive checks, e.g. a balance that cannot be read without the prior balance;
  - a data-path bug in which probe files overwrote evaluation traces;
  - tokenizer false-positive warnings, checked token by token.
- **Seeds.** Single-seed pilots can mislead: the A-vs-B reversal. Claims are therefore conditioned on 3-seed pooling, and method pilots were stopped rather than over-interpreted.

## 7. Limitations
- **One environment family.** BFCL, with 8 API classes and 73 decisive tasks. The held-out-class sets are small (9–35 tasks after conditioning).
- **Second environment small and simulator-free.** τ²-bench airline zero-shot: 13 diagnostic events; forced-prefix episodes without the official user simulator; fault-free completion only 0.15. The check rate, not success, is informative there.
- **Coverage.** Mostly one model family (Qwen3). Method studies are single-seed by design, being stopped at futility.
- **Synthetic faults.** They are injected at the first state-changing call. Real outages are messier.

## 8. Related work
*Every reference marked [verify] must have its exact title, authors and venue checked before submission.*

**Tool-use benchmarks.**
- **Function-calling benchmarks.** The Berkeley Function Calling Leaderboard (BFCL; its multi-turn v4 split is our base environment) [verify] and earlier API-use suites such as API-Bank and ToolBench / ToolLLM [verify] measure whether an agent selects and parameterises the right calls.
- **Stateful benchmarks.** τ-bench and τ²-bench [verify] grade database end states in customer-service domains; AppWorld [verify] and ToolSandbox [verify] add stateful, multi-app execution.
- **What they lack.** These benchmarks evaluate agents on (mostly) reliable tools. Where faults appear, the correct response is unambiguous.
- **Our contribution.** A fault class in which the *same observation* calls for opposite actions, depending on hidden state. A blind policy can at best score 0.5, so recovery must be grounded in checking state.

**Error recovery and corrective training for agents.**
- Step-level preference methods train agents on verified corrective decisions at critical steps: CSO [verify]; Skill-CDPO (Findings of EMNLP 2026) [verify].
- We reproduce a CSO-style verified-DPO baseline and replicate it over three seeds (§4). Its remaining headroom on standard faults is small (about +0.03), while it leaves uncertain-outcome faults untouched (0.51).
- **Our training analysis complements this line of work:** we ask *what transfers* (checking, reading, risk recognition) rather than proposing a better preference objective. We find DPO variants regress ordinary behaviour where SFT does not.

**Idempotency and exactly-once semantics.**
- Distributed systems treat a timed-out request as having an unknown outcome. The standard remedies are idempotency keys (e.g. in payment APIs [verify]) and read-after-write verification. Exactly-once delivery is impossible in general without such mechanisms [verify, e.g. a standard distributed-systems text].
- Our benchmark operationalises this textbook hazard for language-model agents. Our findings show that agents do not adopt the standard remedy on their own.

**Tool metadata.**
- The Model Context Protocol specification defines tool annotations such as `readOnlyHint`, `idempotentHint` and `destructiveHint` [verify: MCP specification revision that introduced annotations].
- We derive these annotations by execution and find that exposing them does not make a fine-tuned 4B agent's recovery rule transfer (§5). In τ²-bench, the risky call's documentation does not mention its side effect at all.

**Synthetic environments for agent training.**
- Procedurally generated tools and environments are increasingly used to scale agent training data [verify: representative works].
- Our synthetic-API results add a controlled finding: diversity over surface form (names, verbs, formats) improves transfer of a recovery rule. But fixed verb–semantics pairings in the generator produce verb-keyed shortcuts that fail on a real API.

**Benchmark documentation.** We follow the Datasheets for Datasets template (Gebru et al.) [verify] (`paper/datasheet.md`).

## Reproducibility
- **Code:** `clean-anchored-agent-training/` and `corrective-training-study/`.
- **Protocols and results:**
  - `docs/headroom_ambiguous_commit.md`;
  - `uncertain_outcome_study.md`;
  - `method_outcome_twins.md`;
  - `method_egr.md`;
  - `method_iar.md`;
  - `method_iar_annotations.md`.
- **Archives:** in `pod_backups/`, with sha256 checksums.
