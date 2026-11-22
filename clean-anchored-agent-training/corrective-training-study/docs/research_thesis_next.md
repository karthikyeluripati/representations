# Next research thesis and novelty assessment (2026-11-22)
**Companion documents:**
- `caw_v2_audit.md` (evidence);
- `semantic_counterfactual_protocol.md` (frozen test).

## 1. Thesis (falsifiable; not yet supported)
> Post-training on execution-verified recovery data teaches small tool agents a portable *check-then-branch* reflex, not a decision policy conditioned on side effects and observability. Under minimal counterfactuals this shows up as one of three behaviours:
> - **(A)** with the API name held fixed, flipping a truthfully documented side effect from repeat-safe to not repeat-safe does not change the recovery decision when no verifier can decide;
> - **(B)** with the side effect held fixed, renaming the API does not change the decision (expected to hold);
> - **(d)** an error-shaped "not recorded" verifier observation is treated as a failure (escalate), not as evidence of a lost write.

**What falsifies it.** The thesis is false if CAW-v2 passes M1–M3 of the protocol (§6 there). In that case the method direction stops.

## 2. What the current evidence supports (`caw_v2_audit.md`)
**Supported:**
- CAW-v2 checks after uncertain writes, transfers that behaviour zero-shot to τ²-bench, and conditions its *post-check* action on the observation:
  - τ²: re-issues the write after 3% of applied-twin checks vs 65–69% of lost-twin checks;
  - BFCL: 17–26% vs 91–93%.

**Not supported:**
- **End-to-end recovery gains on τ²:** end-state +0.058 [−0.038, 0.173] vs SYN64.
- **No-cost claims:** fault-free S0 on the BFCL test split is −0.050 [−0.108, −0.005] vs B, masked by the composite metric.
- **Any claim about side-effect or observability conditioning:** no existing evaluation item makes those factors decision-relevant.

**Seed variability** is large, and two seeds cannot estimate it.

## 3. Novelty audit
Every source in the table was checked against its primary page (arXiv abstract page or repository) on 2026-11-22. Full PDFs were not read unless noted.

| work | what it does | trains models? | overlap with us |
|---|---|---|---|
| **RetryBench**, Liu, Liu, Lu, arXiv 2610.15884 (v4, Nov 2026) | Scores retry decisions after tool timeouts on *retry permission* (an idempotency-key guarantee or proof of non-execution), duplicates and omissions, by replaying hidden worlds. Four synthetic SE tasks, 1,728 decisions, GPT-5.x prompting. Has a weak-evidence `not_found` condition; no true no-verifier condition. | no | **High** for the problem framing and for "permission vs outcome" scoring. Does not vary side effect independently of name; no training; no BFCL/τ² transfer. |
| **Verified Tool Calls**, Kalappurackal Mansoor, Phadke, Rana, arXiv 2608.02645 (Jul 2026) | A wrapper with postcondition verification, verify-before-retry and idempotency keys; 2 simulated workflows, 300 runs, Gemini Flash-Lite. Its three-valued verifier (Unknown) is not evaluated. | no | **High** for the method: CAW-v2 is essentially this wrapper's rule distilled into weights by SFT. |
| **IdempotencyBench** / "Exactly Once Across Retries and Restarts", S. Gopnal Swamy, Zenodo doi 10.5281/zenodo.24165778 (arXiv forthcoming; GitHub `gssanjana4/idempotencybench`) | 320 deterministic tasks; Idempotency Violation Rate under timeout-after-commit, five retry modes, keys and receipts; Claude models evaluated. "Exactly-once behavior is a property of the execution substrate, not the model." | no | **High** for duplicate-effect measurement. No renaming or observability crossing. |
| **PIPE**, Gu et al., arXiv 2602.01611 (Feb 2026) | Interface rewrites that keep semantics fixed; shows trajectory-SFT amplifies interface shortcutting. | yes (trajectory-SFT, analysed) | **Medium**: this is our intervention B (renaming invariance) in general agent tasks. It does not vary side effects. |
| **Same Error, Different Recovery**, Woo, Conde, Reviriego, arXiv 2611.11581 (Nov 2026) | How error *delivery* changes retry / correct / stop behaviour; 13 open models. | no | Low–medium: wording effects; nothing on commit uncertainty. |
| ToolMaze 2606.05806; Beyond Function Calling 2606.25819; ToolMisuseBench 2604.01508; ToolFailBench 2607.04686; AgentCheck 2607.11098; COVERT 2604.09813 | Tool-failure / recovery benchmarks and wrappers; verifiable synthetic tool data | mostly no | Low–medium. Seen only in search summaries: **not verified**, do not cite until read. |

**None of the verified works:**
- (i) varies a tool's documented side effect independently of its name;
- (ii) crosses side effect with verifier observability (decisive / inconclusive / none);
- (iii) trains a model on such counterfactuals and tests held-out semantic combinations.

That gap is real but narrow, and a reviewer would see it as a natural next step after RetryBench and PIPE. The idea of "counterfactual minimal pairs in preference training" is broad and has prior art I have **not** audited. It must be searched before any objective is called new.

**Consequence for the current paper.** `paper/references.bib` cites none of RetryBench, Verified Tool Calls or IdempotencyBench. A submission without them would be rejected on novelty grounds, so they must be added and discussed whatever happens next.

## 4. What kind of paper this can be
- **A. Benchmark/evaluation paper.** Viable, but no longer novel on its own:
  - **Distinctive:** outcome twins embedded in two established benchmarks with state-equality grading; pre-registered training studies; the semantic-counterfactual suite.
  - **Requirement:** it must be positioned against RetryBench and IdempotencyBench.
- **B. Empirical generalization paper.** Possible *if* the protocol yields a clear result in either direction:
  - "SFT teaches a check reflex that transfers across environments but not across side-effect semantics";
  - or the reverse.

  This is currently the most promising framing.
- **C. Methodological main-track paper.** **Not supported.**
  - CAW/CAW-v2 is ordinary SFT of a known wrapper rule.
  - A method claim needs a measured residual gap that matched counterfactual SFT does not close (protocol §7), which has not been tested.
- **Current status: between A and B.** It is insufficient for C, and the main-track *method* framing should be withdrawn for now.

## 5. Recommended next experiment (single)
**Controlled evaluation of the existing policies** on the frozen diagnostic (protocol §4), with **no training**:

| arm | runs |
|---|---|
| π_B | 1 |
| π_B + prompt hint | 1 |
| SYN64 | 3 seeds |
| CAW | 2 seeds |
| CAW-v2 | 2 seeds |
| deterministic contract policy (CPU) | ceiling |

- **Size:** 9 GPU runs × 1,920 episodes (20 held-out families).
- **Resources:** one H100, about 1–1.5 h, **≈ $5 (cap $10)**. All adapters are already on the network volume.
- **Stopping rule:** exactly protocol §6. Every possible outcome maps to one decision (stop / revise / matched SFT pilot), fixed before any model is run.

## 6. Decision
**PROCEED TO CONTROLLED EVALUATION**, pending your authorization for ≈ $5 of GPU time.
- The CPU prototype is built and validated (16 tests).
- No model has been evaluated on it.
- A matched SFT pilot is **not** yet justified, because no gap has been measured.
