# Novelty matrix: PSPO (Paired Solvability Policy Optimization), verified 2026-11-18

**Sources.** arXiv HTML was read for Bench2Robust (2608.11977v2), Sentry (2610.02994), AgentAbstain (2607.10059) and Agentic Abstention (2606.28733v3). Fission-GRPO (ACL 2026), MOSAIC (ICML 2026) and Sawarni et al. (AISTATS 2026) were read from abstracts only; their PDFs were not machine-readable here. ToolMaze is a GitHub repository with **no license** and is not used as code.

| Work | Problem | Parameter updates? | Abstention rewarded in training? | Matched solvable/unsolvable pairs? | Contrastive objective across pairs? | Runtime tool failures? | Evaluates false abstention? |
|---|---|---|---|---|---|---|---|
| Bench2Robust (2608.11977) | Recovery under S1 retry / S2 switch / S3 impossible | Yes (DAPO + KL; curriculum mixes S1/S2/S3) | **No.** "the reward contains no positive term for correct abstention"; S3 is a training condition only | Scenario per episode; matched instantiation not stated | No | Yes (9 noise types; deterministic blocking) | **No.** Held-out evaluation is "solvable settings" only; S3 is unscored |
| Bayesian Tool Memory (same paper) | Beta-posterior recoverability per (tool, error) from training rollouts, put in the prompt with rules (retry / switch / escalate) | No (inference-time context) | – | – | – | Yes | No |
| AgentAbstain (2607.10059, NeurIPS 2026 D&B) | Do agents know when not to act? | **No training** (benchmark) | – | **Yes** (T+, T− differ by one perturbation; Paired Accuracy). Pairs "proposed" as preference data, **not tested** | No | S6 Critical Tool Failure (+ S5, S7, S8) | Implicitly, via paired accuracy |
| Agentic Abstention (2606.28733, NeurIPS 2026) | When to stop instead of act (POMDP: ANSWER / ABSTAIN / ACT) | **No** (CONVOLVE is "without updating model parameters") | – | 1:1 solvable/unsolvable from the same instructions (WebShop) | No | Missing targets / prerequisites, not tool faults | Over-abstention discussed; timely recall |
| Fission-GRPO (ACL 2026) | Recovery from execution errors in small models | Yes (GRPO + fine-tuned Error Simulator; failed trajectories fissioned into recovery rollouts) | Not stated (abstract) | Not stated | Not stated | Yes (BFCL v4 multi-turn, τ/τ²) | Not stated |
| Sentry (2610.02994) | Test-time recovery via a failure playbook | No (playbook only) | No | Counterfactual forks for **evaluation** | No | Progress / reasoning failures | No |
| MOSAIC (ICML 2026) | Safety: act or refuse in multi-step tool use | Yes (preference RL, pairwise trajectory comparisons) | Refusal is a first-class action (reward details unread) | Preference pairs, not matched environments | Preference pairs (safety, not solvability) | Adversarial tool feedback | Benign performance "preserved" |
| Policy Learning with Abstention (AISTATS 2026) | Statistical policy learning with an additive abstention reward | Statistical learner | Yes (additive) | – | – | No (not agents) | – |

## What is already done (we may not claim it)
- An abstention/escalation action, and rewarding correct abstention (Sawarni; MOSAIC refusal; standard).
- Training on tool failures and on-policy recovery rollouts (Fission-GRPO, Bench2Robust).
- S1/S2/S3 scenarios in training (Bench2Robust).
- Paired solvable/unsolvable *evaluation* and paired accuracy (AgentAbstain, Agentic Abstention).
- Recoverability statistics in context (BTM).

## Candidate PSPO distinction (to be tested, not assumed)
A **training objective defined over matched decision points from matched environments**: two histories with an identical prefix up to a decision point t*, under different hidden recoverability z.
- If the histories are **observationally identical** at t* (no evidence yet), the objective pushes the policy toward *information-gathering* actions that separate the worlds (e.g. one retry or one alternative-tool probe), rather than toward the action favoured by the majority world.
- Once the histories **diverge observationally**, it pushes the two decisions apart in the direction of the verified best action per world (continue/switch vs escalate).
- The training signal is the *paired* comparison of log-probabilities at matched points, with weights given by observational identifiability. It is not an episode-level reward.

**Closest alternatives that could already achieve the same:**
- B6, GRPO with a correct-abstention reward on the same S1/S2/S3 mixture.
- B7, SFT/DPO on the same paired examples. DPO on AgentAbstain-style pairs is exactly what AgentAbstain proposes but does not test.

If B6 or B7 match PSPO, there is no method contribution.

## Narrow unresolved limitation (independent of the PSPO outcome)
- **No published method reports trained-policy correct-escalation rate AND false-abstention rate on held-out S3/S1/S2 pairs.** Bench2Robust explicitly leaves S3 unscored.
- That measurement is itself a valid narrow contribution only if training changes it.
