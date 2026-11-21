# Exact prior art (read 2026-11-19)

## How the papers were read
| paper | method sections | results tables | released code |
|---|---|---|---|
| PAFT | full text (arXiv HTML v1) | full text | inspected |
| Fission-GRPO | full text (arXiv HTML v2) | full text | inspected |
| Agent Error Dataset | Sec. 5.3 and App. A, verbatim | Figure 6 / App. X.2 numbers **not** read | none released |
| CSO | Sec. 3–4.1, extracted from the PDF | **not** read | not inspected |
| DART | not read | not read | none linked |
| RFT-forgetting | Sec. 5.1 ablation text only | not read | not inspected |
| Agent-STAR | full text (arXiv 2603.21972 HTML) | full text | inspected |

- **CSO, DART and RFT-forgetting** need a full read before any paper claim.
- **Pinned commits:** see `results/audit/sources.json`.

## What each method optimizes
| work | training signal | objective | protects existing behaviour? | code / licence |
|---|---|---|---|---|
| **PAFT** (arXiv 2607.01084, ICML 2026) | clean trajectories ∪ α=0.3 LLM-generated perturbations: EFP (observation anomaly → supervised corrective action), SBP (fatal error → refusal), SRP (renamed or noisy tool names and docs) | plain NLL (Eq. 1), the same loss as SFT; "differs from SFT only in the training dataset" | only implicitly, through the (1−α) clean mixture; no KL, anchor or regularizer; perturbations are **not verified** (code only checks JSON shape) | OpenAgent @769692c, MIT. Data generation needs an OpenAI key, and **the POI database is not released** (only a 446-byte sample CSV) |
| **Fission-GRPO** (ACL 2026; arXiv 2601.15625) | on-policy RL; failed rollouts get diagnostic feedback from a fine-tuned Qwen3-32B simulator and are resampled G′ times from [x; τ_err; f] | GRPO with −β·KL(π‖π_ref) in both stages (β not reported); reward = format + correctness F1 + length | KL term plus interleaved standard GRPO; correct trajectories are not replayed or anchored | @ebcf553, **no licence**. Simulator weights not released; the launcher defaults to gpt-4.1 feedback. 8×H800 |
| **CSO**, Verified Critical Step Optimization (Findings ACL 2026) | failed policy trajectories → a PRM proposes candidate critical steps → expert alternatives → the **policy** continues each branch → keep only alternatives that reach a correct outcome | DPO on (s_t, a+, a−) at verified steps (Eq. 3), plus an iterative variant with π_ref = π_{i−1} | only DPO's implicit reference term | kiaia/CSO @55e10b4, no licence. CK-Pro-8B, GAIA-Text-103, XBench-DeepSearch |
| **Agent Error Dataset** (arXiv 2609.40111 v2) | success-only SFT vs preventive, action-only and reflective repair SFT on Qwen3-8B | NLL; action-DPO pilot "establishes no recovery benefit" | not studied | not yet released |
| **DART** (EMNLP 2026) | agentic RL | **abstract only:** separate LoRA adapters for reasoning and tool-use tokens with hard token routing, motivated by near-orthogonal or misaligned gradients | parameter separation | no code link |
| **RFT mitigates forgetting** (ICML 2026, PMLR v306) | continual post-training, SFT vs GRPO (Qwen2.5-VL-7B) | GRPO, β=0.01; removing KL causes no degradation (Sec. 5.1) | attributes retention to RFT's *selective (on-policy) update*, not to KL; proposes rollout-based instance filtering | zhhvvv/rft_vs_sft, no licence |
| **Agent-STAR** (arXiv 2603.21972, EMNLP 2026) | SFT then GRPO on TravelPlanner | GRPO, KL removed, clip-high | none | @14aa665, no licence; full fine-tuning, 8×A100 |
| **PCGrad** (Yu et al. 2020); **A-GEM** (Chaudhry et al. 2019) | — | gradient projection | the mechanism itself | standard |

## The Agent Error Dataset observation, as published
- **Direction.** Repair-containing recipes score higher on WebShop-lite and lower on TextQuest (Figure 6). The action-only and reflective WebShop gains and all TextQuest losses survive Holm adjustment, "conditional on one training seed".
- **Caveat** (verbatim): "The single-seed recipes differ in task pools, exposure and optimizer updates; their contrasts measure the combined recipe, not repair supervision alone."
- **Use here:** this is a *motivation* for matched comparisons, not evidence that anchoring helps. We did not read the exact per-environment numbers.

## What CACRT adds beyond each, and what remains a risk
| vs | components shared | what CACRT adds | novelty risk |
|---|---|---|---|
| CSO | outcome-verified branching from failed states; DPO at the divergent step; policy continuations (GPU stage) | (i) the states are *real API errors* produced by controlled, executable error-induction, not PRM-selected; (ii) **counterfactual-twin clean anchors** | **high** for A+B, which are CSO. Only (ii) can be a contribution |
| PAFT | perturbation-style corrective supervision; a clean mixture | anchors are paired per correction and verified by execution; preference loss instead of NLL | moderate: a reviewer can call twins "PAFT's clean data, paired". EXP1 (same states, balanced SFT) is the control for that objection |
| Fission-GRPO | learning from real errors; KL to reference | offline, cheap (no 32B simulator, no RL); explicit retention anchors | low overlap in mechanism; the comparison would need a scaled-down reimplementation |
| DART / PCGrad / A-GEM / KL-retention | gradient-conflict framing | anchors are chosen by observation-counterfactuals; projection (A-GEM) is only an ablation | high if the gain only appears with projection, since that would be A-GEM |
| contrast sets and counterfactual data augmentation (Kaushik et al. 2020; Gardner et al. 2020) | minimal-pair construction | used as a *retention constraint during corrective preference learning* in executable agents | moderate. This needs a full literature check before submission; not done here |

**Bottom line.** Components A and B are CSO and must be described that way. Component C, counterfactual-twin anchoring, is the only thing that could be new, and it is unproven. It also stands or falls on EXP3 > B7_replay and EXP3 > EXP1 at matched budget.
