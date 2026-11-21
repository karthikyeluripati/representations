# Benchmark protocol: WebState paired transitions (2026-11-18)

## Why not BrowserGym (yet)
- **The plan was BrowserGym MiniWoB++,** the smallest real option. It runs MiniWoB tasks as HTML pages in
  Chromium through Playwright (`pip install browsergym-miniwob`, a pinned MiniWoB++ clone, then
  `playwright install chromium`).
- **This machine can't run it:**
  - the C: drive had 0–66 MB free during this session;
  - about 0.4 GB RAM is free;
  - there is no GPU.
- **Most MiniWoB tasks are single-page and fully observable.** Hidden-state variations would have to be
  injected by editing task HTML/JS anyway.
- **We therefore built WebState,** a pure-Python executable simulator with the same observation format as
  BrowserGym's accessibility-tree text. It is an **exploratory proxy**, not BrowserGym, and no result here
  should be read as a BrowserGym result.

## Environment (`scripts/strong_paper/webstate.py`)
**Four apps (families), each a few pages:**
- **settings:**
  - draft vs saved notification setting;
  - a hidden `role`, visible only on the Account page;
  - a hidden `quota_full` that makes Save fail.
- **shop:**
  - cart and coupon;
  - a hidden `member` that changes the price shown on the Product page;
  - a hidden `in_stock`, revealed only at checkout.
- **mail:**
  - a delayed "send later" that fires 3 steps after scheduling;
  - a hidden `offline` that makes sending fail.
- **ticket:**
  - an approval with a hidden delay of 2 or 4 steps;
  - a hidden `can_escalate`, visible on the Profile page.

**Mechanics:**
- **Deterministic:** given hidden variables and an action sequence, the run is fully determined, and every
  reset is reproducible.
- **Observation:** an accessibility-tree-like text with a page title, `[role] label: value` lines, and a
  transient `[status]` toast.
- **Actions:** clicks on `button`, `link` or `checkbox` elements. Unavailable actions raise an error
  (tested).
- **Committal actions** (with side effects: save, checkout, send, submit, escalate, delete, logout) are
  listed per family. They are used only by the "no-commit" probe baseline.

**Cosmetic render operators** (changing the text, not the state):
- **layout:** element order is shuffled.
- **decor:** irrelevant `[static]` lines are added. Train and test families use disjoint banner text.
- **style:** `{class=…}` attributes are added.
- **rename:** synonym labels and titles, two variants per element. This is a **held-out operator**: it is
  never used for tuning or training.

## Ground truth (privileged; used only for labels)
- **H-step trace equivalence (H = 3):** two states are equivalent if every action sequence of length ≤ 3
  produces identical render-independent observation sequences, with identical available actions at every
  step. This is verified by exhaustive execution.
  - Equivalence is never inferred from one transition.
  - 1-step equivalence is recorded separately.
  - The minimal divergence horizon (≤ 6) and a witness action sequence are recorded for every
    non-equivalent pair.
- **Identifiability:**
  - For each item, enumerate every hidden assignment that reproduces its observable history exactly.
  - A pair is **determined** if the equivalence label is the same for every consistent pair of completions.
    Otherwise it is **ambiguous**: no function of the observable history can label it correctly in all
    completions.
  - For ambiguous pairs, the Bayes error floor under a uniform prior is min(p, 1 − p).

## Pairs (`scripts/strong_paper/novelty_reliability.py`)
- **Histories:** per family, 400 distinct random-walk histories (length 0–6, hidden variables uniform).
- **Set A, false-novelty candidates:**
  - **A_render_{layout, decor, style, rename}:** the same history rendered twice (120 items per family).
  - **A_route_{determined, ambiguous}:** two different histories with the same raw current observation that
    are H = 3 equivalent. These are equivalent routes, revert sequences, and so on.
- **Set B, false-equivalence candidates (B_{determined, ambiguous}):** two histories with the **identical raw
  current observation** that are not H = 3 equivalent.
  - Identical history + identical behavior pairs are dropped as trivial.
- **Caps:** at most 25 pairs per group of identical current observations.
- **Splits:**
  - train families settings and shop: 70% train / 30% dev, split by observation-group cluster;
  - test families mail and ticket, held out entirely;
  - the rename operator is held out everywhere.
- **Counts, seed 0, test split:**
  - A_route_determined 66; A_route_ambiguous 94;
  - B_determined 129; B_ambiguous 126;
  - 240 per render operator.
  - Full counts are in `results/strong_paper/reliability.json` under `counts`.
- **Clusters for bootstrap:** the observation group (route and B pairs) or the history (render pairs). The
  bootstrap is task-clustered with 500 resamples.

## Tests (`tests/test_strong_paper_webstate.py`, 8 pass)
- deterministic rendering;
- cosmetic operators preserve state and change text;
- a verified false-equivalence example (unsaved vs saved draft, revealed by logout) is "determined";
- an identical-history hidden-variable pair is "ambiguous", and every deployment key maps it identically;
- an equivalent navigation route is 5-step equivalent but breaks history keys;
- the delayed effect fires on schedule;
- hidden variable names never appear in rendered text, except `role`, which is a deliberately visible field;
- invalid actions are rejected.

## Commands
```
python scripts/strong_paper/novelty_reliability.py --out results/strong_paper --seed 0
python scripts/strong_paper/novelty_reliability.py --out results/strong_paper/seed1 --seed 1
python -m pytest -q -p no:cacheprovider tests/test_strong_paper_webstate.py
```
- Runtime is about 55 s per seed on the laptop CPU.
- `pairs_index.jsonl` stores each pair as (hidden, semantic action ids, render config), so every
  observation can be regenerated.
