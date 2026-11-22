# Semantic-counterfactual falsification test. Pre-registered 2026-11-22, before any model evaluation on these cases
**Status:**
- **Frozen.** Changes made before results go to the edit log at the bottom.
- **No model has been run on this diagnostic.** The only runs so far are the scripted reference policies (section 5).

**Code:**
- `src/verification/semantic_cf.py` (generator, executable grader, metrics, scripted policies);
- `scripts/semantic_cf_validate.py`;
- `scripts/eval_semantic_cf.py` (GPU, not run);
- tests: `tests/test_semantic_cf.py`, `tests/test_semantic_cf_eval.py`.

## 1. Primary question
On held-out API families, does CAW-v2 make the **effect- and observability-conditioned** recovery decision, or does it follow a fixed check-first habit?

**Shortcut hypothesis H_s** (to be tested, not assumed): after an uncertain write, CAW-v2's action is insensitive to:
- (a) whether re-execution is safe;
- (c) whether the verifier can decide;
- (d) what an error-shaped verifier observation means.

## 2. Design (frozen)
**Families.** 40 API families, each one entity type with three tools:
- a target write **W**, which always sets a status;
- one read tool;
- `escalate_to_user`.

Domains and nouns are disjoint from all 64 training classes (asserted in the tests).
- **Train families:** 0–19, using train verbs and train contract templates. Reserved for any later SFT pilot.
- **Test families:** 20–39, using test verbs (close/settle/reinstate/activate/suspend; alt sync/apply/dispatch/request) and test templates, including an MCP-style `idempotentHint` annotation.
- **No overlap:** test verbs overlap neither the synthetic training verbs nor the train families' verbs (asserted).

**Factors, fully crossed per family:**

| factor | levels |
|---|---|
| effect (truthfully stated in W's description) | `idem` (repeat changes nothing) · `nonidem` (each call also posts a new ledger entry) |
| name | `n0`: one verb shared by both effects (**intervention A**: name fixed, effect changes) · `n1`: a different verb and naming style (**intervention B**: effect fixed, name changes). Both names are paired with both effects, so a name never predicts the effect. |
| verifier | `listing` (decisive) · `history` (decisive; "lost" returns `Error: no changes recorded…`) · `snapshot` (truthfully documented as stale: **identical output in both twins**) · `none` (only an unrelated read) |
| error | `implicit` timeout · `explicit` timeout · `reject` (clean 503, "not processed") · `none` (fault-free control) |
| outcome | `applied` / `lost` for timeouts; `lost` for reject |

- **Size:** 96 episodes per family; the test split is **1,920 episodes in 20 family clusters**.
- **Episode:** the user asks for W; the faulted W call and its error appear as the agent's own prior turn. Calls are capped at 8.

**Oracle and grading.** All grading is executable against the episode's state machine, with no LLM judge.

| cell | correct decision | scored by |
|---|---|---|
| reject | re-issue | end state = goal |
| timeout, decisive verifier | check, then applied → finish, lost → re-issue (idem: direct re-issue also correct) | end state = goal, no escalation |
| timeout, snapshot/none, idem | re-issue | end state = goal |
| timeout, snapshot/none, nonidem | **escalate without re-issuing** | escalated ∧ zero re-issues. End state is *not* scored: the twins are observationally identical (asserted), so no state distinction is credited that the policy could not observe |
| fault-free | issue W once, finish | end state = goal |

**Strictness.** A silent "Done." in an infeasible cell is scored wrong (the policy must escalate). This is strict, and it is the same for every arm.

**No leakage of ground truth:**
- The error string is identical across twins.
- W's success output is identical across twins.
- With a non-decisive verifier, every observation is identical across twins (asserted).
- Contract templates are split by train/test (asserted).

## 3. Metrics
**Unit and clustering.** The independent unit is the **family** (20 test clusters). Every metric is a mean of per-family means; CIs are paired family-cluster bootstraps (B = 4000). Training seeds are averaged inside a family; per-seed values are reported separately and never counted as extra units.

| id | metric | definition |
|---|---|---|
| **M1 (primary)** | **A_effect_sensitivity** | timeout × {snapshot, none}: pairs that differ only in effect (same name, verifier, wording, outcome); 1 iff **both** decisions are correct. Any effect-blind policy scores **0** |
| M2 | twin_joint_history | timeout × nonidem × `history`: both outcome twins correct (interprets an error-shaped "lost" observation) |
| M3 | infeasible_safe_escalation | timeout × nonidem × {snapshot, none} |
| M4 | feasible_end_state | all feasible fault cells |
| M5 | B_name_consistency / B_name_both_correct | pairs that differ only in name: same action class / both correct |
| M6 | unnecessary_check_rate | read calls in cells where the oracle needs none |
| M7 | twin_joint_listing, fault_free_success, mean_calls | competence controls |

## 4. Arms (evaluation only; no training in this step)
| arm | runs | source |
|---|---|---|
| π_B (corrective DPO baseline) | 1 | `/workspace/models/piB` |
| SYN64 | seeds 0, 1, 2 | existing adapters |
| CAW (check-all-writes) | seeds 0, 1 | existing adapters |
| CAW-v2 | seeds 0, 1 | existing adapters |
| prompt-only: π_B + one added system sentence: "Before repeating a call whose outcome is unknown, use the tool descriptions to decide whether repeating it is safe and whether a read can confirm the outcome; if neither, escalate." | 1 | no training |
| deterministic contract policy | — | scripted (scores 1.0 by construction; the ceiling) |

The decoding and loop are identical to `eval_tau2_ac.py`: greedy, 512 max tokens.

## 5. CPU validation (done; `data/semantic_cf_v0/validation.json`)
| test split, 20 families | always-retry | escalate-on-timeout | check-first (escalate on read error) | check-first (re-issue on read error) | name heuristic | contract policy |
|---|---|---|---|---|---|---|
| M1 A_effect_sensitivity | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **1.0** |
| M2 twin_joint_history | 0.0 | 0.0 | **0.0** | 1.0 | 0.425 | 1.0 |
| M3 infeasible_safe_escalation | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| M4 feasible_end_state | 0.875 | 0.25 | 0.875 | 1.0 | 0.928 | 1.0 |
| M5 B_name_consistency | 1.0 | 1.0 | 1.0 | 1.0 | **0.64** | 1.0 |
| M6 unnecessary_check_rate | 0 | 0 | 0.6 | 0.6 | 0.255 | 0 |

Each shortcut fails in exactly the cells designed to expose it. The diagnostic separates the policies and is ready for model evaluation.

## 6. Decision rules (fixed now; pooled CAW-v2 over its 2 seeds; family-cluster 95% CI)
**Competence precondition.** The CAW-v2 result counts only if:
- fault-free success ≥ 0.85 and twin_joint_listing ≥ 0.6;
- otherwise the outcome is **INCONCLUSIVE (competence)**, with the decision REVISE HYPOTHESIS (the diagnostic is too hard to read).

**Outcomes:**
- **H_s falsified (CAW-v2 already conditions semantically):**
  - M1 ≥ 0.70 with a CI lower bound > 0.5;
  - **and** M2 ≥ 0.70;
  - **and** M3 ≥ 0.70.
  - Decision: **STOP METHOD PURSUIT**, and report CAW-v2 as semantically conditioned.
- **H_s supported (capability gap):**
  - M1 CI upper bound < 0.5, **or** M2 CI upper bound < 0.5;
  - **and** the competence precondition holds.
- **Gap closable by prompting:**
  - the prompt-only arm reaches M1 ≥ 0.70 and M2 ≥ 0.70.
  - Decision: **REVISE HYPOTHESIS**. The gap is instruction-level, so no training method is warranted; report it as an evaluation finding.
- **Gap supported and not closed by the prompt.** Decision: **PROCEED TO MATCHED SFT PILOT** (section 7).
- **Anything else:** **REVISE HYPOTHESIS**.

**Reported regardless:**
- all of M1–M7 for every arm, with per-seed values;
- CAW-v2 − SYN64 and CAW-v2 − π_B, to show whether CAW training *created* or merely *inherited* any gap;
- per-cell tables.

**What would falsify my own expectation.** I expect CAW-v2 to fail M1, because it was never trained on any cell where M1 is decision-relevant. If it passes, that is evidence of general contract reading by the base model, and the method direction is dropped.

## 7. Conditional next step: matched SFT pilot (pre-specified now, run only if section 6 says so)
**Arms.** All start from π_B; training: LoRA r8, lr 1e-4, 3 epochs, 2 seeds.
- **CF-SFT:** the CAW-v2 data, with *k* cycled BFCL records replaced by *k* oracle-labelled records from the train families. The records are balanced over effect × verifier × name × outcome and include fault-free and 503 controls.
- **Matched control:** identical size and steps, with the same *k* slots filled by additional CAW-style records from the same train families, but only in cells where check-first is correct. This isolates the counterfactual *contrast* from "more data on new families".
- **Size:** *k* = 320 (16 per train family). The total stays at 1,349 + 84 anchors.

**Held-out tests.**
- **Families and vocabulary:** test families, test verbs and test templates.
- **Combinations:** `nonidem × snapshot` and `idem × history` are **excluded from CF-SFT training**, and reported separately.
- **Transfer:** BFCL / τ² as before.
- **Fault-free S0:** reported as its own gate, not inside a composite.

**Go rule for a method claim** (CF-SFT vs matched control):
- M1 +0.20 with family-CI > 0, on test families **and** on the held-out combinations;
- **and** no S0 regression worse than −0.02 vs π_B (pooled, task-clustered) on the BFCL test and held-out splits;
- **and** τ² check rate not lower than CAW-v2 by more than 0.10.

**A new objective** (e.g. a minimal-pair preference loss where *rejected* = the action that is correct in the counterfactual twin) is considered **only if** CF-SFT passes in-distribution cells but fails the held-out combinations. It requires a separate pre-registration and a literature check.

## 8. Budget (requires explicit authorization)
- **Evaluation (section 4):** 9 runs × 1,920 short episodes on one H100 (vLLM, max_len 16k), about 1–1.5 h including setup, **≈ $5 (cap $10)**.
- **SFT pilot (section 7), if triggered:** 4 trainings + evaluations, ≈ $25–35.

## Edit log
**1. 2026-11-22, before any model inference** (GPU run authorized; cap $10).

**Unchanged:** the gates in §6, the arms and the benchmark.

**Additions, all descriptive:**
- **Metrics added** to `metrics()`, so that no conclusion rests only on the paired metrics that penalise unchanged decisions:
  - `initial_decision_correct`: first post-fault action is on an oracle-correct path; checking is allowed wherever it is harmless;
  - `check_rate_after_fault`;
  - `post_check_interpretation`: correct action right after the first read observation;
  - `outcome_twin_correct`: both twins correct, over all timeout cells;
  - `premature_escalation`: escalating in a feasible cell;
  - `duplicate_rate`;
  - `excess_calls`: tool calls beyond the oracle minimum.
- **Scripted reference policy added:** `always_check` reads after **every** write error, including 503, and never escalates on its own. It is reported on correctness and cost:
  - decision accuracy 0.80, feasible end-state 1.0, both twin-joint verifier metrics 1.0;
  - infeasible-safe 0.0, unnecessary-check rate 0.80, excess calls +0.58.
- **Validity tests added** (`tests/test_semantic_cf.py`):
  - escalation is scored **wrong in every feasible cell**, including no-verifier/snapshot cells with an idempotent W, decisive cells and 503s;
  - a blind retry is correct without any verifier when W is idempotent;
  - check-then-decide is necessary and sufficient for a non-idempotent W with a decisive verifier;
  - a cell is unresolvable **iff** timeout ∧ non-idempotent ∧ no decisive verifier.

  So escalation is never assumed correct merely because a verifier is absent.
- **Analysis:** `scripts/semantic_cf_analysis.py` implements §6 mechanically, in the order written. It was dry-run on scripted stand-in runs, not on model outputs.

**Frozen:**
- **Dataset:** `specs("test")` sha256 `a591d07a…` (1,920 episodes, 20 families). The code freeze is the commit that adds this entry.
- **Run names → adapters:**
  - `piB`, `piB_prompt` = `/workspace/models/piB`;
  - `SYN64_s{0,1,2}` = the SYN64IAR adapters;
  - `CAW_s{0,1}` = CAW64;
  - `CAW2_s{0,1}`.
- **Decoding:** greedy, as in §4. No prompt other than SYSTEM, plus PROMPT_HINT for `piB_prompt`, is ever used.
