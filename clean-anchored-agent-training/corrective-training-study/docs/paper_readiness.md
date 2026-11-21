# Paper readiness: honest status

**Status: not ready.** Baselines A and B are untrained, and there is one environment and one seed.

## Gaps, in the order they must close
1. **Train A and B** (`scripts/run_baselines_h100.sh`, about 4.3 H100-h) and measure what failure remains.
2. **Independent environment.**
   - OpenAgent is not usable: the data is unreleased and the rewriter needs a paid API.
   - Recommended: **τ²-bench** (MIT; database-state grading; function-calling format; real tool errors such as invalid order or user IDs). It needs a user simulator, run as a **local open-weight model** (for example Qwen3-8B on the same H100) to avoid paid APIs. This changes the simulator from the official GPT-4.1, so scores are comparable across our arms but **not** with the public leaderboard.
   - Alternative: **AppWorld** (Apache-2.0; state-based unit-test grading; no simulator). But the agent writes code, and a 4B model is likely near the floor.
   - Not implemented yet. Estimate: about 1 day of CPU integration plus about 1 H100-h per evaluated checkpoint.
3. **CSO-faithful pair source.** Mine π₀'s *own* failed trajectories at the first divergence from the executable reference, with the policy continuing both branches. This removes the "induced states" deviation. It is not implemented yet: about 0.5 day CPU, about 0.5 H100-h.
4. **Seeds and statistics.** At least 3 training seeds per arm, pooled paired cluster bootstrap. Held-out has 49 base tasks, so only differences of roughly ≥ 10 points (overall) are resolvable.
5. **Published comparators.** Skill-CDPO-style weighted DPO (cheap, runs on the same pairs) and PAFT-style SFT (B4 in the parent project). Fission-GRPO only as a qualified fixed-feedback reimplementation.

## Target venue (set 2026-11-20): NeurIPS 2027, main track
- **Why NeurIPS 2027:** it is top-tier, and its deadline (projected mid-May 2027; the official call is not yet posted) leaves about 6 months. ICML 2027 (full papers due 22 Jan 2027) is too close for a multi-seed, multi-environment method study done properly. Use ICML only if every bar below is met by early January.
- **The bar, the checklist every result is judged against:**
  1. **Problem and benchmark.** Uncertain-outcome faults:
     - realistic stakes (duplicate bookings, orders, payments);
     - validated state graders (scripted verify = 1.0, blind retry = 0.5);
     - an observation-shift wording (para) and the realistic implicit wording.
  2. **Strong baselines, all beaten:**
     - prompting (std, check);
     - SFT and CSO-style verified DPO (A, B);
     - larger models (8B, 30B-A3B);
     - a trivial "always verify after any error" policy, which must not be the answer.
  3. **Method gain:**
     - significant over the best baseline, at least 3 seeds, paired cluster bootstrap;
     - no regression on S0, S1, S2, S3, RP or RA (retry must stay correct where it is correct).
  4. **Generalization:**
     - unseen functions, API classes and wordings, on a split fixed before training;
     - **at least one second environment** beyond BFCL (candidates: τ²-bench, AppWorld, ToolSandbox);
     - at least 2 model sizes or families.
  5. **Analysis:** ablations of each method component, and the mechanism (when the agent verifies vs retries).
  6. **Reproducibility:** pre-registered protocols (already practised), released code, data and graders.
- **Status against the bar:**
  - (1) mostly met;
  - (2) in progress (uncertain-outcome study running);
  - (3)–(5) not started;
  - (6) on track.
