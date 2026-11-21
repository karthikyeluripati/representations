# Novelty audit: "When Novelty Is Only Syntax", tool-interface invariance in representation-based exploration (2026-11-18)

The previous project's audit (alignment under exploration) is archived at
`docs/archive/alignment_exploration_2026-11-18/`.

- **Sources.**
  - Read: the arXiv HTML of RepExp, Action-Space Shaping, Productive State-Space Exploration and
    Outcome-based Exploration. Their late appendices were only partly read.
  - Read: the **official code** of RepExp (verl-recipe `rep_exp`, commit acc239f, 2026-11-16) and
    ToolSchemaSpace (commit d266b78, 2026-10-05).
  - Other work is covered from abstracts and search summaries.
  - Paper claims are self-reported.

## Foundational work
| | Contribution | Novelty definition | Representation | Equivalent transitions? | Interface transformations? | Novelty under paraphrase? | Learned / frozen | Baselines, limitations |
|---|---|---|---|---|---|---|---|---|
| **RepExp** (Tuyls, Foster, Krishnamurthy, Ash; arXiv 2510.11686, Oct 2025) | Elliptical bonus for test-time selection and RL post-training. Removes diversity collapse; 2.1–4.1× more sample-efficient than Unlikeliness, 3.2–13.4× than GRPO (self-reported) | Leverage hᵀ(λI + Σⱼ hⱼhⱼᵀ)⁻¹h over the **group of rollouts for one prompt**, centered. λ = 0.01. Sparse random projection to 32 dims (128 for AIME), redrawn each call. β = 0.01. Bonus off when no rollout is correct | Mean of the final-layer state over response tokens, from a **frozen copy of the base model** (the reward-model worker loads `Qwen/Qwen2.5-7B-Instruct`), conditioned on prompt + response | No | No | No | Frozen | GRPO, Unlikeliness. MATH / GSM8K / AIME only, binary verifiable reward. Hurts weak models (0.5B). Reward hacking is listed as future work |
| **Action-Space Shaping / ToolSchemaSpace** (Liu, Tan, Wang, Guo; arXiv 2609.34971, 28 Sep 2026; v2 16 Oct) | "Schema bias": success ranges 0–97% across functionally equivalent tool schemas, over 11 LLMs and 32 variants. RL on variant mixtures repairs only the variants it sees | n/a (no exploration) | Tool schema shown to the model | **Yes**: every variant decodes to the same native actions, and lossless validation reaches the same states | **Yes**: 9 operators (merge, split, nest, rename, strip, reorder, transaction, reference, discovery) | No | n/a | Studies *task success* under schema variation, not exploration rewards. Mostly one synthetic environment |
| **Productive State-Space Exploration** (Dupty et al.; arXiv 2610.25957, 19 Oct 2026) | Penalty −γ when the next observation is already in the episode's observation set. ALFWorld and WebShop | ΔI = log\|s ∪ {o}\| − log\|s\|, observations compared by **exact string match** | Observations (text) | Partly: duplicate observations are penalized | No | No (notes that embedding similarity is "needed only in environments with free-form or noisy observations") | None | PPO / RLOO / GRPO / GiGPO baselines. No RepExp or embedding-novelty baseline |
| **Outcome-based Exploration** (Song, Kempe, Munos; arXiv 2509.06941, Sep 2025) | UCB bonus on **final-answer** counts for RLVR | b = min(1, 1/√N(x, a)) over final answers; batch version penalizes repeated answers | Final answer (a canonical outcome) | Yes, implicitly: different traces with the same answer share a count | No | No | None | Math only. Motivates outcome canonicalization because trace space is intractable |

## Related work found
- **EMPO² / exploratory memory-augmented LLM agent** (ICLR 2026, arXiv 2602.23008):
  - state-novelty intrinsic reward, reportedly from embedding similarity to stored states (not verified in
    text);
  - ScienceWorld and WebShop.
- **Mu et al., L-NovelD** (arXiv 2202.08938) and **FoMoRL** (arXiv 2211.04878):
  - novelty over language abstractions or foundation-model embeddings of states;
  - motivated by making novelty *more* semantic;
  - grid environments.
- **Classical invariance theory:**
  - noisy-TV / stochastic-distractor problems with curiosity;
  - bisimulation metrics and Deep Bisimulation for Control (Zhang et al. 2021);
  - state abstraction (Li, Walsh, Littman 2006).
  - All of these establish that novelty should be measured over behaviorally relevant structure, and that
    representations can be made invariant to task-irrelevant variation.
- **Semantic entropy** (Kuhn et al. 2023; Farquhar et al. 2024): clustering generations by meaning before
  measuring diversity. This is the same principle for uncertainty.
- **Strategy-level diversity in RL for LLMs:** uniqueness-aware RL (arXiv 2601.08763; clusters strategies
  with an LLM judge) and DIVER (ICLR 2026).
- **Exploration bonuses in agent RL:**
  - the pass@(k, T) analysis (arXiv 2604.14877): a retrieved-set novelty bonus is not needed;
  - DPEPO (arXiv 2604.24320);
  - DORA Explorer (arXiv 2604.17244).

## Has anyone shown representation-based exploration rewards are sensitive to behaviorally equivalent tool schemas?
**No paper we found measures it.**

- ToolSchemaSpace studies equivalent schemas, but for task success only.
- RepExp studies representation bonuses, but on math and code answers only.
- Productive State-Space Exploration and Outcome-based Exploration *sidestep* the issue by computing novelty on
  canonical objects: exact observations and final answers.
- The principle that novelty should be invariant to behaviorally irrelevant variation is old (noisy TV,
  bisimulation, state abstraction, semantic entropy). Only the empirical instance (LM-representation
  bonuses × executable tool interfaces) is open.

**A structural fact from RepExp's code narrows the question:**
- The covariance is built per prompt group (`uid`), from the rollouts of one prompt.
- Every rollout in a group shares one tool schema, so **renaming the catalog across prompts never enters a
  faithful RepExp bonus**.
- Schema syntax can matter only through:
  1. variation within one schema: equivalent alias tools in the same catalog, argument key order, call
     order of interchangeable calls, natural-language text around the calls;
  2. novelty memories that persist across prompts (RepExp's `persist_covariance` option, or episodic and
     global embedding memories).
- The benchmark covers both regimes (R1, R2).

## Gate A: **PASS (narrow, empirical)**
- **Remaining question:** in the published RepExp configuration and in embedding-novelty baselines, do
  behaviorally equivalent tool-call responses receive novelty comparable to genuinely new behaviors? Does
  this change the selection decisions they drive?
- **Caveat recorded before any result:**
  - in executable tool environments, the canonical transition is available for free after execution
    (decode → native action → state);
  - so simple canonicalization (count or outcome bonuses, as in Outcome-based Exploration and Productive
    State-Space Exploration) is the obvious fix;
  - a learned invariant representation (Gate E) needs a case canonicalization can't handle.

## Sources
- https://arxiv.org/abs/2510.11686 · https://github.com/verl-project/verl-recipe/tree/main/rep_exp
- https://arxiv.org/abs/2609.34971 · https://github.com/williamLyh/ToolSchemaSpace
- https://arxiv.org/html/2610.25957
- https://arxiv.org/html/2509.06941v1
- https://arxiv.org/abs/2602.23008 · https://proceedings.iclr.cc/paper_files/paper/2026/hash/b589d92785e39486e978fa273d0dc343-Abstract-Conference.html
- https://arxiv.org/abs/2202.08938 · https://arxiv.org/pdf/2211.04878
- https://arxiv.org/pdf/2601.08763 · https://arxiv.org/pdf/2509.26209 · https://arxiv.org/pdf/2604.14877
- https://arxiv.org/pdf/2604.24320 · https://arxiv.org/pdf/2604.17244
