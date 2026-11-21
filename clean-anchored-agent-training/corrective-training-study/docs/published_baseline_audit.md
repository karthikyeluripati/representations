# Published-baseline audit (2026-11-19)

## How each source was read
| source | what was read | not read |
|---|---|---|
| CSO repository (kiaia/CSO @55e10b4) | README, SETUP_GUIDE, `run_verify_and_generate_dpo.sh`, the logic of `verify_and_generate_dpo.py` | — |
| CSO paper (Findings ACL 2026) | §3 method, §4.1 setup, from the PDF | results tables |
| Skill-CDPO (Findings EMNLP 2026) | §3.2 objective and training details, from the PDF | experiments in full |
| PAFT, Fission-GRPO, Agent-STAR, Agent Error Dataset | the earlier reads summarized in `../docs/exact_prior_art.md` | — |

## CSO: what the official pipeline actually does
1. **Collect trajectories.** Run the policy (CK-Pro-8B, served by vLLM) on tasks and keep the **failed** trajectories.
2. **Score steps.** At each step, resample alternatives and score each candidate with a process reward model (PRM). The PRM is a prompted closed model behind Azure OpenAI / OpenAI / Anthropic APIs.
3. **Find candidate critical steps** (`find_candidate_pairs_for_verification`): the original action scores **< BAD_MAX=0.5** and the best alternative scores **> GOOD_MIN=0.6**.
4. **Verify** (`continue_from_step_with_candidate`). Execute the chosen candidate and continue with the policy for at most MAX_ADDITIONAL_STEPS=10. Keep the pair only if the final answer is correct. There is **one** continuation, and **only the chosen branch is executed**: the rejected action is the original failing step and is not re-verified.
5. **Pair format.** LLaMA-Factory DPO. The whole step input is **flattened into one human message**, and chosen/rejected are the thought+code text.
6. **Training command (README):** `llamafactory-cli train --stage dpo ... --per_device_train_batch_size 4 --learning_rate 5e-6 --num_train_epochs 3`. Everything else is a LLaMA-Factory default: LoRA (r 8), pref_beta 0.1, sigmoid loss, and a frozen reference = the initial model.
7. **Licence.** The README says Apache-2.0, but the repository has no LICENSE file and the GitHub API reports none.

## Our CSO-style adaptation for executable BFCL
| CSO component | official | ours (Baseline B) | deviation |
|---|---|---|---|
| failure states | the policy's failed trajectories | 252 TRAIN states ending in a **real BFCL error**, produced by controlled error induction. π₀ gets the first action wrong at these states in 422/1,048 samples | states are induced, not mined from π₀'s own failed runs |
| alternatives | PRM-scored expert samples | ground-truth repair action (an executable oracle) | no PRM or paid API; the expert is an oracle |
| negatives | the policy's original action | π₀-sampled actions (48) plus outage responses (retry 123, alternative endpoint 144, escalate 252) | most negatives are candidate actions, not π₀'s own action |
| verification | 1 continuation of the chosen branch | **4** π₀ continuations of **both** branches from identical replayed states; keep only if the one-sided 95% lower bound of Δsuccess > 0 | stricter. All 567 pairs also satisfy the official rule |
| objective | DPO (sigmoid), frozen reference | TRL `DPOTrainer`, sigmoid, β 0.1, reference = frozen π₀ | same |
| optimization | lr 5e-6, batch 4, 3 epochs, LoRA r 8 | the same; lr is also tried at 5e-5 on **dev** | adds a dev lr check |
| prompt format | history flattened into one human message | the model's own chat template (tools in system, multi-turn) | ours matches inference |

**Claim scope:** "CSO-style verified corrective DPO on executable BFCL". It is **not** a reproduction of CSO's GAIA/XBench results.

## Skill-CDPO (Findings EMNLP 2026): closest prior art to the default "next" candidate
- **Critical steps** are found by rollout divergence between an expert and the local policy, with n = m = 4 rollouts per group and threshold ε = 0.3.
- **Preference pairs** use **all n×m cross-group rollouts** at those steps.
- **Objective (Eq. 11):** L = −E_t [ (Δ_t^α / Ẑ) · (1/nm) Σ_ij q_t,ij · log σ(β log-ratio(a*) − β log-ratio(a°)) ].
  - The step weight is the normalized criticality, which averages to 1 over the critical steps.
  - The pair weight q is the score gap.
- **Training:** β 0.5, lr 5e-6, 3 epochs.
- **Implication:** weighting corrective pairs by measured evidence is **already published**, as criticality- and score-gap-weighted step DPO. Any "evidence-conditioned selection or weighting" we propose has to beat a Skill-CDPO-style weighting as a direct baseline.

## Comparators that are not runnable as published
| method | blocker |
|---|---|
| Fission-GRPO | the error-simulator weights are not released, the launcher defaults to gpt-4.1 (a paid API), it was trained on 8×H800, and there is no licence. A fixed-feedback GRPO adaptation would be a qualified comparison only |
| PAFT / OpenAgent | the POI database is not released (only a 446-byte sample), and the rewriter needs a paid API. PAFT-style perturbation SFT can be run on BFCL as the parent project's B4 arm, but the OpenAgent shifts cannot be evaluated |
