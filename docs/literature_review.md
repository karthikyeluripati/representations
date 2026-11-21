# Literature review and novelty-risk assessment

Last updated: 2026-11-17. **Verification level:** I confirmed arXiv IDs, titles and authors for every entry.
I read the method details for RepExp, ExTra and DiRL from the arXiv HTML pages; a few hyperparameters came
from a fetch-tool summary and are marked † (check them in the paper before quoting). Rows marked "abstract-level"
were checked only against abstracts and search snippets. Their reported numbers are **not** verified.

## 1. Requested papers

| | RepExp | ExTra | DiRL |
|---|---|---|---|
| Reference | Tuyls, Foster, Krishnamurthy, Ash. *Representation-Based Exploration for Language Models: From Test-Time to Post-Training*. arXiv:2510.11686 (ICLR 2026) | Hu et al. *ExTra: Exploratory Trajectory Optimization for LM RL*. arXiv:2606.24994 | Xia et al. *Reasoning or Memorization? Direction-Aware Diversity Exploration in LLM RL*. arXiv:2606.10346 |
| Representation | Policy's **last-layer hidden states, mean-pooled over response tokens** (beats last-token pooling). Sparse random projection: d=512 at inference; d=32† for RL, redrawn each step | External sentence embedder (all-MiniLM-L6-v2†); no policy hidden states | Policy's final-layer hidden state. A fixed "reasoning − memorization" direction *k* (difference of means over contrastive prompt sets) weights per-token gradient features |
| Exploration incentive | **Elliptical bonus** h^T Σ^{-1} h, with Σ = λI + Σ_j h_j h_j^T. Inference: greedy coreset selection among N sampled responses (Sherman–Morrison updates, λ=1†). RL: reward bonus β·bonus, Σ reset per group, β=0.01† | Reward 1[correct]·(1 − max cos to a memory of correct responses) added *after* GRPO normalisation; plus entropy-guided prefix regeneration | Cosine-novelty score over Φ features, rewarded if the response lies on the "reasoning" side of *k* and penalised on the "memorization" side |
| Learns task-progress / value signal? | **No** (stated: no auxiliary models) | **No** (mean token entropy is an unsupervised heuristic; MC prefix values used only to validate it, ρ ≈ −0.23) | **No** (verifier reward; GPT-4o labels used once to build contrastive sets) |
| Inference-time vs RL | **Both** | RL only | RL only |
| Domains | Single-turn: MATH, GSM8K, MBPP+, Game-of-24, AIME; proteins | Single-turn math (Qwen3-1.7B, Nemotron-1.5B) | Single-turn math/reasoning (Qwen3-1.7B/4B-Base) |
| Agents / ALFWorld? | No | No | No |
| Code | rep-exp.github.io (project page) | github.com/allen4747/extra | anonymous review link only |
| Author-stated limitations | Weak models may see no gain or degrade; high temperature gives "novel but incoherent" responses; token-level variant costly | Math only; embedding novelty is the costliest component | Single fixed direction; depends on the contrastive data |

**ALFWorld** (Shridhar et al., ICLR 2021; github.com/alfworld/alfworld, v0.4.2 installed here). It is a text
environment on TextWorld's PDDL engine: 6 task families, splits train (3553 solvable games),
valid_seen (140) and valid_unseen (134), and the environment supplies the admissible commands.

**AgentGym** (Xi et al., arXiv:2406.04151; github.com/WooooDyy/AgentGym). It serves 14 environments over HTTP,
**including ALFWorld**, and provides AgentTraj-L trajectories plus the AgentEvol self-evolution method. The
project page does not describe a value or progress model for AgentEvol, and I did not read the paper, so that
question is **unverified**.

## 2. Closest prior work on the *decision-aware* side (found during the novelty check)

These matter more for novelty than the three requested papers. All rows are abstract-level.

| Work | What it does | Overlap with this project |
|---|---|---|
| **QLASS** (Lin et al., arXiv:2502.02584; github.com/Rafa-zy/QLASS) | Builds exploration trees with an SFT agent, estimates per-step Q-values, trains a Q-network, and at inference picks the sampled action with the highest predicted Q. Evaluated on **ALFWorld**, WebShop and SciWorld | **Directly covers the "value-only selection" arm**, on the same benchmark |
| Step-level Q-value models (Zhai et al., AAAI 2025) | MCTS-annotated step Q-values → Q-value model used to select actions | Same as above |
| AgentPRM, *Process Reward Models for LLM Agents* (arXiv:2502.10325) | Per-step PRM ≈ Q-function trained on Monte Carlo rollout returns; ALFWorld, 3B models | **Covers MC-rollout labelling of state–action values** (our Phase 2 dataset) |
| IPR, *Watch Every Step!* (arXiv:2406.11176) | MC-estimated step rewards for agent training; ALFWorld | MC step values |
| AgentRM (arXiv:2502.18407) | State values from rollouts, used as a reward model for generalisation | MC values / reward models |
| *Productive State-Space Exploration for Multi-turn RL in LLM Agents* (arXiv:2610.25957) | Intrinsic penalty for actions that yield no new informative observation; ALFWorld, WebShop | **Close to our "wasted action" notion and to novelty shaping for agents** |
| EMPG (arXiv:2509.09265), SELAUR (arXiv:2602.21158), T²PO (arXiv:2605.02178) | Entropy- and uncertainty-shaped rewards or exploration control for multi-turn agent RL (ALFWorld, WebShop) | **Uncertainty-guided exploration for agents** (policy-entropy based, not epistemic uncertainty over a value) |

## 3. Novelty-risk assessment

| Candidate contribution | Risk | Reason |
|---|---|---|
| Value-guided action selection in ALFWorld | **Very high: already done** | QLASS, step-level Q models, AgentPRM |
| MC-rollout labelling of step / action values | **Very high: already done** | AgentPRM, IPR, AgentRM, QLASS |
| `value + β·uncertainty + λ·novelty` score | **High** | A UCB-style combination of known parts. The research plan already treats it as a heuristic, not a contribution |
| Hidden-state elliptical novelty applied to *agent actions* | Medium-high | A direct transfer of RepExp to a new domain. Incremental unless something non-obvious appears |
| Uncertainty / novelty shaping for agent RL | High | EMPG, SELAUR, T²PO, 2610.25957 |
| **Measuring** whether representation-space novelty predicts counterfactual action usefulness in multi-step agents, with **verified state branching**, matched compute and clustered CIs | **Lowest (moderate)** | I found no paper that directly tests "novel ≠ useful" for agent actions with causal branch outcomes. This is the defensible core. It is an empirical or diagnostic contribution, and a negative result counts too |
| Showing *when* novelty bonuses waste interactions (e.g. novel-but-irrelevant actions) and whether value-aware representations fix it | Moderate | Depends on the effect being real and not already explained by QLASS-style value models |

**Implications for the plan:**
- QLASS should be the "value-only" reference point, or at least be cited as one.
- The project should aim at the measurement question, and not at a new scoring rule.
- "Decision-aware representations beat novelty" is only interesting if it holds against a *value-only* baseline
  at matched compute. If value-only already captures the gain, the representation story adds nothing.

## 4. Method fidelity in this repository

- `elliptical_novelty` is a **RepExp-inspired adaptation**, not RepExp. It applies the elliptical bonus per step
  to candidate *actions*, with Σ accumulated over embeddings of previously selected actions within an episode.
  RepExp selects among complete responses for a verifier, or uses the bonus as a group-level RL reward.
  There is no random projection (H=896 for Qwen2.5-0.5B) and no per-step centring.
- `cosine_novelty` is a generic max-min cosine-distance bonus. It is **not** RepExp.
- A faithful RepExp reproduction would need best-of-N selection over whole sampled trajectories (or RL with
  group-level bonuses). That needs the GPU budget, so it is deferred.
