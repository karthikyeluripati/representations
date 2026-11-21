# Information acquisition in long-horizon language agents: novelty audit (2026-11-18)

Question under review: can language agents tell uncertainty that is worth resolving from uncertainty that
should be ignored, and acquire information only when it pays for its cost?

Sources were read from arXiv / ACL Anthology / PMLR abstract and HTML pages. MetaAct's PDF could not be
text-extracted, so it is covered from its abstract only. Paper claims are self-reported and not reproduced
here.

## Core papers

| | Problem | Method / objective | Entropy / IG / VOI / RM | Explicit action cost | RL post-training | Benchmarks, models | Limitations | Overlap with us |
|---|---|---|---|---|---|---|---|---|
| **Calibrate-Then-Act** (Ding, Tomlin, Durrett; arXiv 2602.16699, Feb 2026) | When to stop exploring and commit, given that each step costs | Separates calibration from action: the LLM is given an inferred **prior over the hidden environment state** and reasons about the cost-uncertainty tradeoff. The gain persists under RL of both the baseline and CTA | Explicit prior / expected value vs cost. A Pandora's-box-style VOI | **Yes, central** | Optional; gain preserved under RL | Synthetic Pandora's box, knowledge QA with optional retrieval, simplified coding | Needs a priors model per environment; no long-horizon embodied tasks; the hidden state is small | **High** for "is information worth its cost". It owns the explicit-VOI framing |
| **IGPO** (Wang, Dai, Ye et al.; arXiv 2510.14967, ICLR 2026) | Sparse reward and advantage collapse in multi-turn agents | Turn-level reward = increase in the policy's probability of the **ground-truth answer** after each turn, added to the outcome reward | **Information gain** (intrinsic belief update) | No | Yes (GRPO-style) | Multi-turn search QA (v2 retitled "…Search Agents") | Needs a known answer string to score belief; QA, not embodied | **Medium-high**: IG-as-reward is taken. Not directly applicable to ALFWorld (no answer to score) |
| **Look Before You Leap** (Ye, Shi et al.; arXiv 2605.16143, May 2026) | "Premature exploitation": agents act on priors before gathering environment information | Explore-then-Act: an exploration budget is spent first with a separate verifiable exploration reward (**Exploration Checkpoint Coverage**), interleaved with task rollouts | Coverage reward (not entropy / VOI) | Fixed interaction budget, not per-action cost | Yes | ALFWorld and others; Qwen2.5-7B, Qwen3-4B, Llama3.1-8B | Coverage ≠ usefulness; budget fixed a priori | **High** on embodied information gathering and on reporting looping |
| **Clearing the Fog / SaFaRi** (Guan et al.; arXiv 2608.14339, Aug 2026) | Expert data has ~0 exploratory actions; RL collapses away from exploration | Teacher-generated exploratory trajectories (length-penalized screening r − nγ), then **contrastive pairs whose MC-rollout values differ by a margin** ("productive exploration vs redundant wandering"), trained with DPO + SFT | MC value of actions (5 rollouts); entity-discovery metric for evaluation only | Step penalty in data screening only | Offline DPO (IPR-style) | WebShop, InterCode-SQL, ScienceWorld; Llama-3.1-8B, Mistral-7B; GPT-4o teacher | Heuristic efficiency metric; teacher-bounded | **High**: "productive vs redundant exploration" via counterfactual values is essentially our Stage-3 label |
| **MetaAct** (Tong et al.; Findings IJCNLP-AACL 2026) | Inefficient trial-and-error, redundant actions | Agent predicts a meta-action (FOCUS / GUESS / RESUME / UNDO) before acting; group-based RL with episode- and step-level advantages | Not determinable from the abstract | Not determinable | Yes | ALFWorld, WebShop | (PDF not read) | **Medium**: explicit control of the explore/commit mode on ALFWorld |
| **IG clarification** (Deng et al.; ICML 2026, PMLR 306; arXiv 2606.03135) | When and what to ask the user | Reward = Bayesian belief update toward the ground-truth goal caused by the clarification exchange | **Information gain** | Implicit (+0.3 steps) | Yes | τ-Bench, 5 backbones; +3.7% success | Needs a ground-truth goal; user-in-the-loop | Medium (IG for asking, not for environment actions) |

## Additional related work found
- **Entropy reduction vs tool-call quality** (arXiv 2602.02050, Feb 2026): reports a strong positive
  correlation between entropy reduction and high-quality tool calls. This is the closest prior work to an
  "entropy reduction ↔ useful information" measurement, in a tool-use setting.
- **Align While Search** (CVPR 2026): chooses actions by expected information gain in a task-specific belief
  posterior, including ALFWorld experiments.
- **Entropy-shaped agent RL on ALFWorld**: EMPG (arXiv 2509.09265), EPO (arXiv 2509.22576), AEM
  (arXiv 2605.00425). These use policy entropy for exploration and credit modulation, not to judge
  information value.
- Earlier in this project: QLASS, AgentPRM, 3SPO, BEACON, POISE (state value / process reward).

## Gap assessment
- **Already covered:**
  - cost-aware explore/commit with explicit VOI (CTA);
  - IG rewards when an answer or goal is known (IGPO, Deng);
  - exploration coverage rewards on ALFWorld (Look Before You Leap);
  - telling productive from redundant exploration with MC values (SaFaRi);
  - meta-level explore/commit actions on ALFWorld (MetaAct);
  - entropy-shaped agent RL (EMPG / EPO / AEM).
- A new "cost-aware gather/execute/commit policy" (Stage 5 idea) would be an **incremental combination**
  of CTA + SaFaRi + MetaAct. **No defensible method gap.**
- **One narrow, testable empirical gap remains:**
  - In a long-horizon embodied setting, with **counterfactual branch values at matched states**, does the
    policy's own uncertainty (pre-action entropy, or the entropy reduction an action produces) indicate
    when an information-acquisition action is worth more than the alternatives?
  - IGPO assumes IG is useful; 2602.02050 reports a positive correlation for tool calls; SaFaRi uses MC
    values but no entropy.
  - None tests whether *policy-entropy signals* separate valuable from worthless information gathering
    once matched on state and observables.
  - This would be an **analysis contribution** at best. It justifies only a cheap offline test on data we
    already have, not a method.
- **Recommendation:** run the cheap offline test (Stages 1–3) and STOP the method direction unless entropy
  signals show a robust, held-out, matched-state relationship with information value. See
  `docs/experiment_log.md` for the outcome.

## Outcome of the offline test (see `docs/experiment_log.md`)
- Entropy reduction does not identify task-relevant information. AUROC 0.47 (3B) and 0.67 (7B), vs 0.86–0.92
  for "first time doing this" / step.
- Pure examine / look actions never revealed new relevant information.
- Pre-action entropy does not predict the counterfactual value of information gathering (ρ ≈ 0.06,
  CI ±0.3).
- Entropy × action type adds nothing to logp for ranking.
- **Recommendation: STOP.**

## Sources
- https://arxiv.org/abs/2602.16699
- https://arxiv.org/abs/2510.14967
- https://arxiv.org/abs/2605.16143
- https://arxiv.org/abs/2608.14339
- https://aclanthology.org/2026.findings-aacl.223/
- https://proceedings.mlr.press/v306/deng26l.html
- https://www.arxiv.org/abs/2602.02050
- https://openaccess.thecvf.com/content/CVPR2026/supplemental/Bae_Align_While_Search_CVPR_2026_supplemental.pdf
- https://arxiv.org/html/2509.09265v1
- https://arxiv.org/html/2509.22576v1
- https://arxiv.org/pdf/2605.00425
