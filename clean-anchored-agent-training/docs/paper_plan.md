# Paper plan (conditional; nothing below exists yet)

Write the paper only if the pilot meets the GO rules in `training_protocol.md` §5.

## Claim, if the evidence supports it
- **Main claim.** In corrective preference learning for tool agents, anchoring each verified correction's executed observation-counterfactual twins does two things beyond the alternatives below:
  - it prevents the measured negative transfer (real errors treated as outages, and lost fault-free competence);
  - it improves held-out-family repair, at a matched anchor budget.
- **Alternatives it must beat:**
  - generic replay or KL;
  - gradient projection;
  - balanced SFT on the same states.
- **Framing.** Verification and localization are presented as CSO, not as contributions.

## Full study, after GO
1. **Environments.**
   - BFCL v4 multi-turn with three leave-one-family-out folds (Vehicle, Travel, Trading).
   - A second independent environment. OpenAgent if its data can be obtained; otherwise τ²-bench or another executable benchmark with real API errors (to be chosen and audited).
2. **Models.** Qwen3-4B and one more size (Qwen3-1.7B, or 8B if 4B is positive).
3. **Seeds.** 3 training seeds for every primary arm.
4. **Arms.** B0–B9 plus the ablations:
   - B (unverified);
   - C (whole-trajectory instead of localized): B2 or B3 vs EXP3;
   - D (no anchor): EXP2;
   - E (A-GEM): EXP3_agem;
   - F (same data, balanced SFT): EXP1;
   - G (corrective fraction 0.1 / 0.3 / 0.5).
5. **Published RL comparator.** A scaled-down Fission-GRPO-style corrective GRPO with KL, using real BFCL error feedback instead of the 32B simulator. It is reported as a qualified reimplementation (about 6–8 H100-h per fold).
6. **Mechanism.** Retention curves; gradient-conflict measurements; and interventions (twins vs generic at equal count, and T-out vs T-clean only) rather than cosine plots alone.
7. **Costs.** Environment executions, model calls, tokens and GPU-hours for every arm, including data construction.

## Estimated full-study compute
- About 3 folds × 2 sizes × (8 arms + 4 ablations) × 3 seeds = 216 runs.
- At about 1 H100-h per 4B run, that is about 150–250 H100-h.
- Re-estimate from the pilot's measured throughput.
