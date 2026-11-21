# Experimental protocol (2026-11-18)

**Status: not executed.** The novelty gate failed (`docs/novelty_audit.md`). This file records the smallest
decisive experiment, in case anyone revisits H1 as an empirical note. It is not a GO plan.

## Only experiment worth considering: H1 at test time (no RL)
**Question:** on a real agent pool, does RepExp elliptical selection raise or lower the violation rate compared
with random selection from the same pool, at matched k?

1. **Environment**
   - Reuse an existing verifiable setting instead of building one:
     - the CATCH coding testbed (released code), or
     - a rebuilt LeetCode + `run_tests`-overwrite loophole (2611.09828).
   - Violation = executable predicate over the final repository and the action history (file diff touches a
     forbidden path, or the test harness is redefined).
   - Success = hidden tests run in a clean copy.
   - The checker is the only judge. No learned evaluator anywhere in labeling or evaluation.
2. **Pool**
   - Qwen2.5-7B-Instruct, about 200 tasks × 64 samples at T = 1.0. Record mean-pooled last-layer states, as
     RepExp does.
   - Stratify tasks by the base model's violation rate p̂ (minority vs majority violating).
3. **Arms at k ∈ {4, 8, 16}**
   - random;
   - top-logp;
   - RepExp elliptical selection (official code);
   - elliptical selection restricted to checker-compliant samples (the trivial fix);
   - a checker-based rejection filter.
4. **Metrics**
   - success, violation, compliant success;
   - diversity: distinct strategy clusters, defined by diff-signature clustering independent of the LM states;
   - task-cluster bootstrap with paired differences.
5. **Pre-registered prediction from the simulation**
   - Δviolation > 0 for tasks where p̂ is below the share of the violating strategy cluster, and < 0 above it.
   - If confirmed, H1 reduces to "diversity selection equalizes strategy modes." That is a short note, not a
     paper.
   - A finding would be **unexpected** only if violations rise even when they're already the majority mode, or
     if the compliant-restricted bonus doesn't remove the effect.
6. **Gate to anything further**
   - An unexpected result that survives held-out tasks.
   - Even then, RL-time follow-ups must beat the checker penalty (Shortcuts and Boundaries) and gated diversity
     (MoDA) at a matched budget.
