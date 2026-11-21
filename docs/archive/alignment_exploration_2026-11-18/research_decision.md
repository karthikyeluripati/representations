# Research decision: constraint-aware representations for aligned exploration (2026-11-18)

## Decision: **STOP**
The stopping criterion "already adequately covered by existing research" is met. Implementation and GPU work
were not started.

1. **Novel?**
   - Not as a method.
   - The benchmark and failure mode, with the penalty fix, are Shortcuts and Boundaries (2611.09828's coding
     loophole too).
   - Learned-representation monitors during RL are 2611.09828 and Goodfire 2609.19101.
   - Compliant diversity is quality-gated diversity (MoDA, DIVER).
2. **Closest paper:** *Shortcuts and Boundaries* (arXiv 2611.12865) for the problem. *Designing Effective
   Monitor-Based Interventions* (arXiv 2611.09828) for the representation-as-RL-signal part.
3. **Remaining gap:**
   - Only H1, the effect of RepExp-style exploration on violation rates, is unmeasured.
   - Its sign is largely predictable: elliptical selection equalizes strategy modes, raising violations when
     they're a minority and lowering them when they're the majority.
4. **Demonstrable tension?**
   - Yes, but already demonstrated: task-only RL learns profitable violations, reaching 97.9% under 25%
     conflict exposure (Shortcuts and Boundaries).
   - Diversity-driven amplification isn't separately shown in the literature. Our simulation says it depends on
     the base rate.
5. **Why representations instead of a checker?**
   - No good reason in verifiable settings: the checker is the ground truth and upper-bounds a learned proxy.
   - In transfer settings, learned monitors are studied and are evaded under RL pressure.
6. **Smallest decisive experiment:**
   - A test-time H1 check on CATCH or the `run_tests` loophole: random vs RepExp vs compliance-restricted
     RepExp, stratified by base violation rate, with the checker as the sole judge.
   - About 2–4 H100-hours (`docs/experimental_protocol.md`).
   - **Not recommended:** a positive result confirms a prediction and still doesn't open a method contribution.
7. **GPU now?** No.
8. **Publication potential:**
   - Low at NeurIPS, ICML or ICLR as framed.
   - The H1 measurement could be at most a workshop note on "exploration bonuses and constraint violations."
   - A better-positioned open problem nearby is monitors that stay robust under optimization pressure (probe
     evasion). It's crowded and is a different project.

## DARE context
This is the fourth direction closed on evidence: action ranking, state value, information acquisition, and now
alignment under exploration. The common lesson is that in verifiable agent environments, simple observable
signals or programmatic checkers dominate learned frozen-representation signals.
