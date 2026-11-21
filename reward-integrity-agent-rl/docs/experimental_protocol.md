# Experimental protocol (bounded; NOT authorized, NOT run)

## Hypotheses (tested separately)
- **H1:** RL raises weak acceptance without a matching rise in strong success.
- **H2:** composition errors produce different dynamics from single-output errors.
- **H3:** FP types and frequencies drift on-policy.
- **H4:** budgeted strong feedback beats the best existing defenses at matched audit cost and matched compute.

Current CPU evidence pre-empts H2/H4 (see research_decision.md).

## Stage A: frozen pilot (the only recommended next step)
| Item | Setting |
|---|---|
| Model | Qwen2.5-1.5B-Instruct (pin the revision at download) |
| Inference | vLLM |
| Interface | JSON tool calls, ≤ 40 steps |
| Tasks | Invoice + inventory, seeds 0–199 each; 8 samples at T = 0.8 |
| Metrics | Weak and strong pass rates, FP share of accepted, FP mechanisms before and after the trace check, steps, action diversity |
| Cost | ≈ 3,200 episodes × ~1.5k tokens ≈ 5M tokens; ~10–20 min on one H100 including load (≈ $1) |
| Storage | ~4 GB of weights on the pod volume |

## Stage B: first RL run (only if Stage A shows trace-check-surviving composition FPs)
- **Framework:** verl (multi-turn agent rollout) or TRL GRPO with a custom multi-turn environment. Use an established framework.
- **Algorithm:** GRPO, G = 8, 64 tasks/step, 200 steps, lr 1e-6, KL 0.001.
- **Model:** Qwen2.5-1.5B-Instruct, the same initialization for every arm.
- **Arms:**
  - A: strong reward (clean upper bound);
  - B1: W1 reward;
  - B2: W2 reward;
  - B1 + trace check (cheap deterministic control);
  - single-step noise control (strong + i.i.d. flips matched to W1's FP rate);
  - C: frozen.
- **Defenses** (only if B shows H1): NOPO; LB budgeted audits with reflected penalty (Audit Where Hacks Pay); random audits; selective-control correction. All at matched audit counts and matched rollouts.
- **Evaluation:** every 20 steps on a fixed validation set:
  - seeds 1000–1199 per family;
  - unseen sizes n = 7–8;
  - an unseen workflow family, held out entirely (to build: transfers/balances);
  - the test split, seeds 5000+, untouched.
  - Report: weak and strong acceptance, FP share, mechanisms, trajectory length, skipped reads, entropy, and KL to init. Task-clustered bootstrap CIs.
- **Seeds:** 2 per arm initially.
- **Compute estimate:**

  | Item | Estimate |
  |---|---|
  | Per step | ~512 trajectories × ~700 generated tokens ≈ 0.36M tokens; ~15–25 s generation + ~10–20 s update |
  | Per run | ≈ 2 h for 200 steps |
  | 5 arms × 2 seeds | ≈ 20 H100-h ≈ $50–60 |
  | First decisive subset (A, B1, B1 + trace, 1 seed) | ≈ 6 H100-h ≈ $15–18 |

- **Storage:** checkpoints are LoRA or full 1.5B (≈ 3 GB each). Keep every 50 steps only.
- **Stop rules:**
  - B1 strong success not below A's, and FP share not rising → no H1.
  - B1 + trace check ≈ A → cheap correction suffices.
