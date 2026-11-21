# Novelty audit: "When Novelty Lies" (2026-11-18)

## Scope
- **Question:** can a learned, history-conditioned transition representation give exploration rewards that avoid
  both:
  - **false novelty:** different-looking but behaviorally equivalent situations;
  - **false equivalence:** similar-looking but behaviorally different situations;
  under partial observability, for language agents?
- **How sources were verified:** every source was fetched on 2026-11-18. Items marked [S] come from a model
  summary of a page section that was not read verbatim; they are weaker evidence.
- **Not read in full:**
  - DPEPO and TEB PDFs: the binary PDFs could not be parsed locally (no disk space for tools), so their
    arXiv HTML versions were read instead. Both reads were truncated at about 40k characters.

## Close papers

| | Productive State-Space Exploration (arXiv 2610.25957, Oct 2026) | DPEPO (arXiv 2604.24320; ACL 2026) | Action-Conditioned Bisimulation for GUI Agent Memory, "APSG" (arXiv 2609.38778, Sep 2026) | ScreenSearch (arXiv 2605.16024 v2, Aug 2026) | TEB: Task-Aware Exploration via a Predictive Bisimulation Metric (arXiv 2602.18724; ICML 2026, PMLR 306) | WebStep (arXiv 2606.15673; COLM 2026) |
|---|---|---|---|---|---|---|
| **Contribution** | Duplicate-observation penalty for multi-turn LLM-agent RL (PPO, Qwen2.5-1.5B) | Parallel-environment exploration with diversity rewards | Merge rule for a GUI agent's outcome memory | Uncertainty-aware desktop-OS exploration, screen graph | Visual RL: learned predictive bisimulation metric and a metric-based exploration bonus | Process-level web-agent benchmark |
| **Objective / reward** | r_int = −γ if o_{t+1} ∈ s_t (exact string match) | DAR (action repetition, depth/width) + DTR (repeated s→a transitions); equality of states/actions **not defined** | d_a(s,t): outcome-rate gap + γ·TV of successor-block distributions; merge when d ≤ τ_b; iterated union-find | r = 1[new state] + 1[new edge] + [u(s) − u(s′)]₊; u = outcome dispersion of matched action signatures | Predictive bisimulation (predicted-reward differential + W₁ of a learned transition model); potential-based bonus γΦ(s′) − Φ(s) | No reward; a deterministic semantic MDP "dual" of each site |
| **State** | Set of unique observations so far | Text observations; text history (compressed) | Observation signature; actions as role+label templates | UI Automation element set; hybrid sparse/dense merge (Jaccard ≥ 0.93); typed text kept in the action signature | Latent z = φ(s) from pixels (MetaWorld), raw state (Maze2D) | Positional state + revealed item attributes |
| **Partial observability** | "Partially observable" stated; no model | Not stated | Framed as aliasing; resolved by per-action outcome statistics across episodes | Explicit; handled empirically via outcome dispersion; no belief state | **Assumes an MDP** | Detail-only attributes; "hard negatives" |
| **Learns a representation** | No | No (policy only) | No ("nothing is trained") | No for state identity (pretrained embedding + handcrafted features) | **Yes** | No |
| **Intrinsic exploration reward** | Yes (penalty) | Yes (diversity rewards) | No (ε-greedy detours) | Yes | Yes | No |
| **False novelty** | Acknowledged as missed near-duplicates | No | Can merge different-looking states with matching outcomes; not tested separately | Bounded merge; not framed this way | Implicitly, via the metric (distractors) | No |
| **False equivalence** | No | No | **Central motivation** ("pages that look alike but behave differently") | **Yes**: similar screens with different next states are flagged for probing | Implicitly | Hard-negative items are a related idea |
| **Evaluation** | ALFWorld, WebShop | ALFWorld, ScienceWorld | MiniWoB++ via BrowserGym; 10 tasks; Qwen3-8B frozen; single seed | 11 Windows apps; 5,524 VM-hours | MetaWorld (6 tasks), Maze2D | 10 self-hosted sites, 1,800 tasks |
| **Key result** | 94.0 vs GiGPO 86.7 on ALFWorld (single runs) | 98.2 ALFWorld average; ablations remove DAR/DTR and lose ≤ 2 points [S] | Success 0.782 vs ReAct 0.723 (+0.060, CI [+0.033, +0.090]) | Ambiguity term +8.9% state AUC, trading off edge coverage | Reports gains over ICM, CeSD, etc. [S] | Small agents 34–37% terminal success |
| **Limitations** | Exact match only; small action spaces | Equality undefined | Needs repeated tries of the same action at both states; degrades to identity with element-id actions | Needs repeated trials or replays; a one-step selector | Markov assumption; task-specific η | No exploration reward |

Other relevant work found:
- **Exploration and memory under partial observability:** arXiv 2608.05111 (2026, unreviewed) argues that
  novelty bonuses should be computed over a history statistic (a recurrent state), not raw observations,
  and that the bonus and the memory architecture are coupled.
- **Classical intrinsic motivation:** NGU (episodic novelty, arXiv 2002.06038) and DEIR (discriminative
  episodic reward, arXiv 2304.10770).
- **Bisimulation representation learning:** DBC (Zhang et al. 2021), MICo, and Chen & Pan 2022, on which TEB
  builds.
- **Predictive state representations** and belief-state methods.
- **GUI exploration and deduplication** found by search (abstracts or snippets only):
  - JAMEL (arXiv 2606.01528) uses code coverage as behavioral novelty;
  - KG-Agent (arXiv 2510.15259) adds "functionally similar" edges;
  - UIE-XPlore-Bench (arXiv 2506.17779) uses drift-tolerant novelty tables;
  - WebRLED (arXiv 2504.19237) uses episodic and global novelty.
- **Brief 3 (this repo):** representation novelty is not invariant to tool-interface syntax, and
  canonicalization fixes it.

## Critical question: which limitation would our approach overcome?
The only distinction that holds up:
- **APSG and ScreenSearch** resolve false equivalence with interaction evidence: repeated tries of the same
  action at two states, across episodes or VM replays. Neither learns a representation, and neither gets
  history-identifiable distinctions without that evidence.
- **TEB** learns a behavioral metric but assumes Markov states.
- **PSSE and DPEPO** use exact or undefined equality.
- **The proposal:** a learned, history-conditioned, task-conditioned transition representation that separates
  history-identifiable hidden-state differences without extra probing, while staying invariant to cosmetic
  and route differences.

Why the distinction is weak:
1. **As a method**, it is a recurrent-encoder version of predictive bisimulation (TEB, DBC) applied to text
   observations, i.e. a learned belief-state abstraction. That combination is standard. "History-conditioned
   bisimulation" is not a new objective, and the brief forbids relabelling it.
2. **APSG's interaction-based rule** needs no training and already targets the false-equivalence half on
   MiniWoB++. Our CPU experiment shows that a simple deterministic version (probing non-committal
   actions) solves most identifiable cases (see `failure_analysis.md`).
3. **The novel, defensible part** is the measurement: a joint benchmark of both failure modes with
   identifiability labels (determined vs ambiguous given the observable history). That is an analysis or
   benchmark contribution, not a method.

## Gate A
- **Analysis / benchmark:** PASS (narrow). No paper jointly measures false novelty and false equivalence of
  exploration rewards with identifiability labels.
- **Proposed learned method:** FAIL. It is not distinguishable from recurrent predictive bisimulation plus
  APSG-style interaction evidence.
