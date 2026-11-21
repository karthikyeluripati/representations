# CACRT method specification (v1, written before any training)

## Setting
- **Policy and reference.** π_θ is the policy. Training starts from π₀, the stage-0 policy: Qwen3-4B-Instruct-2507 fine-tuned on fault-free and injected-outage recovery trajectories (the recipe behind the previous project's 38% false-outage behaviour). π_ref = π₀.
- **Decision state.** A state s is a rendered dialogue prefix plus the environment state reached by replaying that prefix's calls. Actions a are single assistant messages (one tool call, or a final reply).
- **Verifier.** V(τ) ∈ {0,1} is BFCL's state-based grader: every involved class instance must equal the ground-truth end state, and the episode must not end in escalation (for S3 it must end in escalation).

## Component A: outcome-verified corrective pairs
- **Corrective states.** Taken from train tasks only, and every one ends in a *real* BFCL error:
  - **RP:** the agent skipped ground-truth call G[j], so a later call G[m] returns a real precondition error.
  - **RA:** a one-character typo in a string argument returns a real error. The correct value must be visible in the context.
- **Validity checks.** A state is kept only if two things hold. First, ignoring the error and continuing fails the grader (the error matters). Second, the full repair passes it.
- **Branching.** For a+ (the first repair action) and every candidate a−, both branches are executed from deep copies of the same replayed checkpoint. The state hash is asserted to be unchanged afterwards, and no continuation is ever spliced across states. Both branches then continue under the **same** continuation policy π_c with the **same** tool budget.
- **Retention rule.** Keep (s, a+, a−) only if Δ = Ê[V | a+] − Ê[V | a−] has a positive one-sided 95% lower bound.
  - With deterministic π_c, one execution per branch is exact.
  - With π_c = π₀ sampled K times, a normal approximation with per-arm variance is used (`verify_stochastic`).
- **CPU stage.**
  - a− ∈ {identical retry, alternative endpoint, escalate}.
  - π_c = an outage-rule surrogate.
  - A competent "repairer" oracle continuation is also recorded. Under it only escalation is outcome-critical, which shows that criticality depends on the continuation policy.
- **GPU stage.** a− = π₀'s own sampled actions at s, and π_c = π₀ (CSO-style).

## Component B: localized preference update
At the verified divergent decision only:

L_DPO(θ) = −E_(s,a+,a−) log σ( β[ (log π_θ(a+|s) − log π_ref(a+|s)) − (log π_θ(a−|s) − log π_ref(a−|s)) ] )

- log π(a|s) is summed over the action's tokens; the prefix gets no gradient.
- This is exactly the CSO objective (Findings ACL 2026, Eq. 3) applied to tool-error states, and it is **not** a contribution.

## Component C: counterfactual-twin clean anchors (the only candidate contribution)
For each retained corrective state s with pivot call c (the call that returned the real error), two *twin* states are built from the same task and executed and verified in the environment:
- **T-out(s).** The correct prefix, with the same call c returning an *injected outage* instead of a real error. The kept action is ã = identical retry, and it is verified to complete the task.
- **T-clean(s).** The correct prefix, with c succeeding. The kept action is ã = the next ground-truth call (or the final reply), and it is verified.

Objective, with D_twin = the twins of the corrective states in the batch's dataset:

L_CACRT(θ) = L_DPO(θ) + λ · E_(s̃,ã)∈D_twin [ −(1/|ã|) log π_θ(ã|s̃) ]

The anchor term is an NLL by default. A KL(π_ref ‖ π_θ) variant is implemented. λ is fixed, at 1.0 in the pilot. On GPU, twins are restricted to those π₀ already gets right greedily: this is the "already-correct behaviour" filter, and the fraction it removes is reported.

**What is new, and what is not.**
- **Not new:** each of the following is standard and appears as its own baseline:
  - anchoring with replay or KL (B7);
  - gradient surgery (B8 PCGrad, and EXP3_agem's A-GEM);
  - verified step-level DPO (CSO, EXP2).
- **Candidate claim:** the *choice of what to anchor*. Each anchor is a minimal observation-counterfactual of a correction, executed and verified in the environment, at the exact decision boundary that the correction risks moving (real error → repair vs outage → retry vs success → proceed).
- **Hypothesis.** Negative transfer from corrective learning concentrates at these boundary twins. If so, anchoring them protects retained behaviour at a far lower anchor budget than generic replay, and transfers better because what is learned is the *error-type distinction*, not a tool-specific rule.

**Falsifiers, all pre-registered** (each is a matched comparison):
1. **Premise test** (`analyze_interference.py --mode premise`, on π₀): cos(∇L_DPO, ∇R_twin) is not more negative than cos(∇L_DPO, ∇R_generic). In that case there is no mechanistic reason for twins.
2. **EXP3 vs B7_replay** (same pairs, same anchor count and λ, generic anchors): no held-out difference means twin selection adds nothing.
3. **EXP3 vs EXP1** (balanced SFT on exactly the same states and actions): no difference means it is "just balanced data".
4. **EXP3 vs EXP2** (no anchor): no retention difference means anchoring is unnecessary.

## Adaptive variant (ablation E)
- **Rule.** Apply A-GEM with the twin-anchor gradient as the constraint: g ← g_DPO − min(0, g_DPO·g_A)/‖g_A‖² · g_A.
- **Status.** This *is* A-GEM, applied to the twin anchors, and is reported as such. No new adaptive rule is proposed unless the fixed-λ results show a measured failure that it fixes.

## Compute overhead vs EXP2
- **Data construction.** +2 verified twin executions per retained state. On CPU this was 8,632 environment tool executions for all train states, pairs and anchors together (manifest).
- **Training.** One extra forward/backward on an anchor batch per step, which is ≈ 1.6–2× EXP2's per-step cost, since anchors are single actions with the same long prefix. B7_replay pays the same cost, so EXP3 vs B7 is compute-matched.
