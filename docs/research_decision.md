# Research decision: "When Novelty Is Only Syntax" (2026-11-18)

The previous project's decision is in `docs/archive/alignment_exploration_2026-11-18/research_decision.md`.

## Gates
| Gate | Result | Evidence |
|---|---|---|
| A: novelty | **PASS (narrow)** | No paper measures LM-representation novelty under equivalent tool interfaces. The principle (noisy TV, bisimulation, semantic entropy) and canonical-object bonuses (final-answer UCB; exact-observation penalty) exist |
| B: verified equivalence | **PASS** | 180 tasks, 12 domains; every response executed in ToolSchemaSpace SimEnv; 0 violations; 10/10 tests |
| C: reward mismatch | **PASS** | Faithful RepExp (7B) gives a surface duplicate exactly the bonus of a new behavior (ratio 1.00). Duplicate recognition 0.43–0.72, where 1.0 is perfect and 0.5 is chance; replicates on 1.5B. Cross-schema duplicates are scored more novel than new behaviors on 96–100% of tasks |
| D: behavioral impact | **PASS (test-time selection only)** | On 5,760 sampled responses, 18–31% of valid pairs are surface duplicates. RepExp selection gets 0.38–0.44 fewer unique states than state novelty, has lower success@4 than random (−0.04 / −0.05), and ties a TF-IDF baseline |
| E: learned-method advantage | **FAIL** (not trained, no headroom) | Canonical novelty (schema decode, free) is within 0.03–0.08 unique states of the oracle. The remaining gap is call order, which the state multiset fixes |

## Answers
1. **Is it novel?**
   - The measurement is new; the principle and the fix aren't.
   - Novelty should be computed over behaviorally relevant structure, which is old in RL. Canonical-outcome
     bonuses already exist for LLMs (arXiv 2509.06941, 2610.25957).
2. **Closest papers:**
   - RepExp (arXiv 2510.11686) for the method under test;
   - Action-Space Shaping / ToolSchemaSpace (arXiv 2609.34971) for equivalent schemas, studied for success
     only;
   - Productive State-Space Exploration (arXiv 2610.25957) for exact-duplicate state penalties in agent RL.
3. **What remains unexplored:**
   - A quantitative account of how LM-representation novelty behaves under behavior-preserving interface
     changes.
   - Its mechanism here: with n = 8 rollouts, a 32-dim projection and λ = 0.01, the leverage bonus
     saturates at about (n−1)/n for any response that isn't an exact copy.
4. **Does tool-interface syntax change representation novelty rewards?**
   - Yes. Equivalent calls get the same bonus as new behaviors, because the published bonus detects only
     exact copies. Different transitions also get the same bonus.
   - Across schemas, renaming moves representations 1.5–4.5× further than changing the action does.
5. **Does it change exploration decisions?**
   - In test-time selection, yes: about 0.4 of 4 slots go to redundant behaviors, and outlier or invalid
     responses are over-selected.
   - In RL, not tested. The bonus's effect on the advantage is about 0.005 against exact copies only.
6. **Does canonicalization solve it?**
   - Yes, nearly perfectly, at zero GPU cost.
   - Decoding to native calls reaches 2.97 / 2.57 unique states against the oracle's 3.00 / 2.65, and the best
     success@4.
7. **Is a learned representation justified?**
   - No. There is no failure case for canonicalization on executable tool interfaces.
   - A learned invariant model could only matter where no decoder or canonical state exists (free-form web or
     GUI observations). That is a different benchmark and a different question.
8. **GPU for the next experiment?** No. The pod is stopped.
9. **Main-track potential?**
   - Low. The headline ("compute novelty on the decoded action, not on the text") is a strong, cheap
     baseline that already beats the method studied. It's not a new algorithm, and a main-track reviewer would
     call the fix obvious.
   - The measurements could make a short empirical or workshop note, or an analysis section in a broader
     paper on exploration bonuses for agents:
     - RepExp's saturation at its published group size;
     - the prevalence of surface duplicates (18–31%);
     - selection losses compared with canonical novelty.

## Decision: **STOP**
- STOP the proposed paper and the Transition-Invariant Exploration method: Gate E fails, and simple
  canonicalization already reaches the oracle.
- No RL post-training.
- The verified findings are preserved in `docs/experiment_log.md` and `results/interface_invariance/`, and could
  support a workshop-length empirical note without further compute.
