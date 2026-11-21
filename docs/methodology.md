# Methodology

## Research question
Can representations that preserve information about **future task success** support more sample-efficient
action selection than representations that reward **behavioural novelty**? The first goal is to find out
whether "useful" and "novel" can be told apart and predicted at all. We do not assume the hypothesis is true.

## Environment
- ALFWorld 0.4.2 text environment, driven through TextWorld 1.7.0 `PddlEnv` directly (`src/dare/environments`).
- We apply the same game filters as ALFWorld's `AlfredTWEnv`: no movable or Sliced tasks, solvable games only.
- Candidate actions are the environment's admissible commands, which every policy receives identically.
  `env.max_candidates` can truncate the set.
- The episode budget is `env.max_steps` (default 50) environment actions.
- Success means the PDDL goal check passes (`won`).

## State branching (verified)
- `PddlEnv.copy()` raises `NotImplementedError`, so branching uses **deterministic replay**: reset the same
  game file and re-apply the action prefix.
- State equivalence is checked on a SHA-1 hash of the full sorted PDDL fact set, and also on the observation
  text and the admissible-command set.
- `scripts/check_environment.py` and `tests/test_alfworld_replay.py` check that replays match at every step,
  that branching twice from the same point gives the same state, and that different actions give different states.
- Cost: about 0.6 s to reset plus about 0.08 s per replayed step on the WSL laptop CPU.

## Policies (matched candidate sets)
| name | selection rule |
|---|---|
| `random` | uniform over candidates |
| `frozen_lm` | argmax (or softmax at temperature T) of the mean log p(action + end-of-turn \| prompt) under a frozen instruct LM |
| `cosine_novelty` | `frozen_lm` score + λ·min_j(1 − cos(h_a, h_j)) over previously chosen action embeddings. Generic, **not RepExp** |
| `elliptical_novelty` | `frozen_lm` score + β·h^T Σ^{-1} h, Σ = reg·I + Σ_j h_j h_j^T. **RepExp-inspired adaptation**, see the literature review §4 |

- The prompt contains: the task intro, the last `max_history` (action, observation) turns, the current
  observation and the admissible list. It uses the chat template.
- Scoring is one prefix forward pass with a KV cache, then one batched pass over the candidates.
- Embeddings: `h_a` is the mean last-layer hidden state over the candidate's tokens, conditioned on the state.
  The context embedding is the last prompt position.
- Token accounting: every prompt token and candidate token processed is logged per step. Generation is 0,
  because policies score rather than generate.

## Logging
- `transitions.jsonl` holds one row per step: task id, split, family, seed, objective, observation, history
  reference, candidates, selected action, policy scores, LM log-probs, novelty bonus, next observation, reward,
  won, done, state fingerprints, wasted flag, tokens, timings, model id and config hash.
- `episodes.jsonl` holds the full action list and the outcome.
- Runs are resumable. Records carry an attempt id, so transitions left by a crashed attempt are dropped.
- `--verify` replays every logged episode and checks all logged fields against the environment.

## Operational definitions
- **Wasted action:** an action after which the PDDL fact-set fingerprint equals a state already visited in
  the episode. This covers no-ops (look, examine, inventory, failed commands) and returns to earlier states.
  Revisiting a location to *use* it later still changes facts, such as which receptacles are open.
  This is a conservative lower bound on waste.
- **Branch value of (s, a):** the success rate over n rollouts of the frozen rollout policy after executing a
  at the verified-restored s. We also record mean remaining actions, mean remaining tokens and the
  Beta(1,1)-posterior std of the success rate ("outcome uncertainty"; this is aleatoric plus finite-sample
  uncertainty, not model-epistemic uncertainty).

## Decision dataset (Phase 2)
- Sources: trajectories on **train-split** tasks only. The script refuses any other split.
- At each sampled decision state, the candidates are the source policy's own choice plus K−1 uniformly sampled
  alternatives. Actions are not labelled "good" just because they lie on a successful trajectory.
- Fit/dev partition is **by task instance**, via a stable hash. One task family (`pick_two_obj_and_place`) is
  held out entirely for a family-transfer test. Final evaluation uses valid_seen and valid_unseen, which are
  disjoint task instances, and unseen rooms for valid_unseen.

## Statistics
- Success rates are reported with Wilson 95% intervals.
- Ranking and calibration metrics are pairwise ranking accuracy, within-state Spearman, AUROC, Brier score and
  ECE (`src/dare/evaluation/metrics.py`, unit-tested against sklearn and scipy).
- CIs come from a percentile bootstrap that resamples **tasks**, since states are clustered within tasks.
- We report both absolute and relative differences, and budgets are matched by candidate set and token accounting.

---

# Methodology: tool-interface invariance in representation-based exploration (2026-11-18)

This project is separate from the DARE sections above, and none of their hypotheses are reused. Data, code
and tests are new:
- `scripts/iface_lib.py`, `construct_paired_transitions.py`, `verify_tool_equivalence.py`,
  `audit_novelty_rewards.py`, `evaluate_exploration.py`;
- `tests/test_interface_invariance.py`.

## Question and gates
Do representation-based exploration rewards give behaviorally equivalent tool-call responses the same
novelty as genuinely new behaviors? Does that change exploration decisions?

| Gate | Question |
|---|---|
| A | Not already answered |
| B | Equivalence verified by execution |
| C | Reward mismatch |
| D | Behavioral impact on sampled responses |
| E | A learned invariant representation beats canonicalization |

## What counts as RepExp here (mandatory distinction)
- **Faithful formula, from the official code** (verl-recipe `rep_exp`, `elliptical_reward_model_worker.py`
  and `train_elliptical.sh`):
  - frozen `Qwen/Qwen2.5-7B-Instruct`, i.e. the same model as the policy, loaded as a separate reward
    worker;
  - `last_hidden_state` averaged over response tokens, conditioned on prompt + response;
  - sparse random projection to 32 dims (`randomize_sparse_matrix=True` → redrawn per call; we average 16
    draws);
  - float64, centered within the prompt group, λ = 0.01;
  - leverage bonus (`reward_type=leverage`, `normalization=none`) over the group of rollouts for one prompt.
  - **Our `repexp` and `repexp_greedy` are this.**
- **Justified adaptation:**
  1. The unit is a tool-call response, not a math solution. RepExp's bonus is defined over any response
     text, so this is legitimate.
  2. Stage 3 groups are *constructed* (oracle-encoded responses), not sampled. Stage 4 uses sampled
     groups.
  3. Inference-time selection uses greedy elliptical selection with Sherman–Morrison updates, as in the
     paper's test-time algorithm. **We did not run the paper's selection code**, which is not in the RL
     recipe.
- **Exploratory proxies, not RepExp:**
  - `lev_full` (no projection), `repexp_lrel*` (λ relative to group scale);
  - `nn_cos_*` (episodic nearest-neighbor cosine novelty);
  - `farthest_*` (max-min distance selection);
  - MiniLM and TF-IDF text novelty;
  - `repexp_noctx` (the same formula on context-free features).

## Novelty measures compared
1. raw text embedding (MiniLM, char-TF-IDF);
2. pretrained LM hidden-state novelty (`nn_cos_lm`, `lev_full`, `repexp_noctx`);
3. faithful RepExp (`repexp`);
4. canonicalized action novelty (`canon` = 1 / count of the decoded ordered native-action sequence);
5. observation / transition novelty: in this environment the observation after a call is a deterministic
   acknowledgment of the executed native action, so this coincides with (6);
6. oracle semantic-transition novelty (`state` = 1 / count of the final native-action multiset).

## Metrics
- **Duplicate recognition:** in a group of 8 holding one equivalent pair and 6 unique behaviors,
  P(bonus of a duplicate member < bonus of a unique member), with ties counted as ½.
  - Oracle = 1. A reward that ignores semantics and treats surface differences as novelty scores ≈ 0.5 or
    lower.
- **Duplicate bonus ratio:** mean bonus of the pair / mean bonus of the unique members. Oracle 0.5.
- **Decision: wasted-slot rate.**
  - Choose k = 7 of the 8 (the number of distinct behaviors) by greedy novelty selection; a slot is wasted
    if both members of the equivalent pair are chosen.
  - Oracle 0; random 0.75.
- **R2:** fraction of tasks where a G-duplicate under another schema is scored less novel than a new
  native behavior, against a native memory; plus raw LM distance ratios.
- **Stage 4 (sampled):**
  - prevalence of surface duplicates (same state, different text);
  - duplicate recognition on natural groups of 8;
  - selection of k = 4 of N = 16: unique valid states, redundant picks, success@k;
  - tokens and compute.
- **Uncertainty:** task-cluster bootstrap (2,000 resamples), per transformation family; per-domain min/max
  for the faithful method.
- **Controls:**
  - token length (correlation of bonus with length);
  - catalog order (R1 vs R1_shuf);
  - permutation of the group (leverage must be invariant);
  - group size (n = 4 vs 8);
  - log-probability of variants vs canonical;
  - λ sensitivity.
