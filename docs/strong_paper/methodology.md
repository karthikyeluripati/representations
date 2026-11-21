# Methodology: "When Novelty Lies" feasibility study (2026-11-18)

## Status of each component
| Component | Status |
|---|---|
| WebState simulator (`scripts/strong_paper/webstate.py`) | Exploratory proxy built by us. Not BrowserGym, not a reproduction of any paper |
| H-step trace equivalence, identifiability labels | Exact (exhaustive execution) |
| Exact observation, string TF-IDF, canonicalization, transition key | Standard baselines, implemented here |
| History set | Faithful to PSSE's state definition (set of observations, exact match), with our canonicalization applied first |
| Field tracker, mutation log | Our own deterministic heuristics: the strongest simple history rules we could write |
| Probe k = 1, 2 (+ no-commit) | **APSG-inspired, not APSG.** It compares canonical observed outcomes of label-matched actions from simulator snapshots of both states. APSG pools empirical outcome statistics across episodes and successor blocks; we don't |
| Learned combiner | Diagnostic only: logistic regression / depth-4 tree over 11 key-agreement and distance features. Not the proposed method |
| Frozen LM embeddings; representation-based bonuses (RepExp) | **Not run.** No GPU; no model weights fit on disk. Brief 3 already measured RepExp on tool calls (pod, 7B) |
| TEB (predictive bisimulation) | **Not run.** No public code, visual RL, MDP assumption. No substitute was labelled TEB |
| Count-based exploration | Our own: per-key UCB-style least-count action choice. Actions are keyed by canonical label (or element id for the oracle) |

## Definitions
- **Item:** (goal, observation texts o_0..o_t, action texts a_0..a_{t−1}), plus the privileged final state,
  which only the labels and the snapshot probes use.
- **Decision rule for "same":** key equality for keys; distance ≤ τ for TF-IDF, with τ tuned on the dev
  split (train families, non-rename) to maximize balanced accuracy.
- **False-novelty rate:** fraction of equivalent pairs judged different.
- **False-equivalence rate:** fraction of non-equivalent pairs judged the same.
- **Bayes floor:** for an ambiguous pair, min(p, 1 − p), where p is the fraction of history-consistent
  completions that are equivalent. It applies jointly over A and B pairs; a single subtype's rate can sit
  below it by sacrificing the other subtype.
- **Probe cost:** environment steps used per pairwise comparison.

## Controls and safeguards
- **Matched inputs:** every method sees the same observable history. Probes additionally get the same
  number of snapshot steps for both states.
- **Held-out data:** whole families (mail, ticket) and one operator (rename), plus disjoint decor text for
  test families.
- **Independence:** cluster bootstrap over observation groups; two pair-sampling seeds.
- **Fixed thresholds:** τ is fixed on dev before the test evaluation, and nothing was re-tuned after seeing
  test results.

## Known weaknesses
- **The simulator is ours.** Hidden variables were chosen to cover the cases of interest, so prevalence is
  meaningless.
- **Probing is optimistic in the proxy:** navigation is reversible and cheap here. In real sites,
  side-effect-free probing can be costlier.
- **Rename** uses fixed synonym lists, so it measures label sensitivity, not natural paraphrase.
- **Exploration** uses tiny apps (≤ 7 pages), so absolute headroom may be larger in real sites.
