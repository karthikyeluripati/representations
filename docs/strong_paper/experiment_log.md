# Experiment log: "When Novelty Lies" (2026-11-18)

- **Earlier project results are untouched:** the previous projects' results and docs (`docs/*.md`,
  `results/interface_invariance/`) were not modified.
- **New work lives only in:**
  - `docs/strong_paper/`, `results/strong_paper/`;
  - `scripts/strong_paper/`;
  - `tests/test_strong_paper_webstate.py`.
- **No git branch was created** (nothing is committed in this repo yet), so the separate directory serves as
  the isolation.

## Stage 0: novelty audit
Sources fetched:
- **Verbatim, from HTML (truncated at about 40k characters):**
  - 2610.25957 (PSSE);
  - 2609.38778 (APSG; full HTML);
  - 2605.16024 (ScreenSearch);
  - 2606.15673 (WebStep);
  - 2602.18724 (TEB);
  - 2604.24320 (DPEPO).
- **Abstract or landing page:** ACL 2026.acl-long.2151 (DPEPO).
- **BrowserGym** README and MiniWoB README.
- **Search snippets only:**
  - 2608.05111;
  - JAMEL (2606.01528);
  - KG-Agent (2510.15259);
  - UIE-XPlore-Bench (2506.17779);
  - WebRLED (2504.19237).

**Outcome:** Gate A passes narrowly for the analysis, and fails for the learned method (see
`novelty_audit.md`).

## Stage 1: environment
- **BrowserGym was rejected for local use:** disk was at 0–66 MB and Chromium was impossible to install.
- **WebState** was built instead: 4 families, hidden variables, delayed effects, and 4 cosmetic operators.
- **Tests:** 8 pass. On the first run, 2 failed because of test bugs:
  - the witness was not unique;
  - `role` is both a hidden variable and a visible field label.

  We fixed the tests, not the simulator.

## Stage 2: baseline reliability
**Command:** `python scripts/strong_paper/novelty_reliability.py --out results/strong_paper --seed 0`.
Outputs: `reliability.json`, `exploration.json`, `pairs_index.jsonl`, `run_seed0.log`.
- **Tuned thresholds:** τ_tfidf_obs = 0.391, τ_tfidf_hist = 0.039.
- **Smoke run:** we added `tracker_fields` and `mutation_log` after the smoke run, as stronger deterministic
  baselines, before the full run. No test numbers had been seen at that point except the smoke sample's.
- **Main test-family results:** in the table in `failure_analysis.md`.
- **Seed 1** (`results/strong_paper/seed1/`, pair index not saved, to conserve disk): same pattern. Key numbers,
  seed 0 → seed 1:

  | Method / subtype | Seed 0 | Seed 1 |
  |---|---|---|
  | history set, A_route_det | 0.77 | 0.80 |
  | mutation log, B_det | 0.14 | 0.09 |
  | no-commit probe, B_det | 0.14 | 0.15 |
  | ambiguous fraction of B | 0.49 | 0.51 |

- **Learned combiner:**
  - Logistic regression coefficients put the largest weight on `hist_full` (1.49) and `d_tfidf_obs`
    (1.78).
  - On held-out families it predicts "different" for every route pair: false novelty 1.00.

## Stage 3: failure analysis
- **Ambiguous fraction:** about half of the hidden-state false equivalences are not identifiable from
  history.
- **Determined pairs:** the best history-only deterministic rule leaves 0.28 balanced error; no-commit
  probing leaves 0.07.
- **Exploration headroom** of a privileged oracle key over the canonical key: 0–7% more unique states.

## Stages 4–6
Not run:
- Gate D was not met;
- Gate E (learned representation) requires a GPU, and no RL is permitted.

## Decision: **STOP** (see `research_decision.md`)
