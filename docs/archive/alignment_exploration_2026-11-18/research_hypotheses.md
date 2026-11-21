# Research hypotheses: alignment under exploration (2026-11-18)

**Alignment, narrowly:** following explicit, task-specific user constraints while achieving the intended
objective. Nothing broader is claimed.

These hypotheses are separate from the DARE hypotheses (action ranking, state value, information acquisition),
which are closed.

| | Claim | What would support it | Status after the audit |
|---|---|---|---|
| **H1** (empirical) | Representation-based exploration (a RepExp-style elliptical bonus) raises behavioral diversity *and* constraint violations, compared with ordinary stochastic exploration at a matched budget | Paired increase in violation rate at matched diversity and success, across seeds and held-out tasks | **Unmeasured in the literature, but largely predictable.** The sign depends on whether violating strategies are a minority or majority mode (simulation, `docs/experiment_log.md`). At best a short empirical note |
| **H2** (representation) | Representations trained on compliant vs violating trajectories carry alignment structure that transfers to unseen tasks, constraint types and paraphrases | Held-out AUROC above a plain classifier, an LLM judge and a lexical baseline using the same labels | **Covered.** Probe and judge monitors are studied (arXiv 2609.19101, 2611.09828, PROBE). Separability alone isn't a contribution |
| **H3** (intervention) | Constraint-aware representations improve the trade-off between success, compliant diversity and violation in exploration and RL, beyond penalties, shields, rejection sampling and gated diversity | A matched-budget RL comparison beating the checker penalty and a gated-diversity baseline | **Fails.** In verifiable settings the checker penalty is the ground truth and upper-bounds a learned proxy (Shortcuts and Boundaries). Learned monitors during RL get evaded (2611.09828). Gated diversity is MoDA / DIVER |

H2 doesn't imply H3: a probe with high held-out accuracy can still be evaded once it's optimized against (2611.09828).

## Why a learned representation instead of a constraint checker?
- **When a checker exists (the verifiable case the brief requires for evaluation):** use the checker. A learned
  representation adds error and an attack surface.
- **When a checker doesn't exist:** ground truth for evaluating the representation also doesn't exist. The brief
  forbids using the same learned evaluator as label and judge.
- The only coherent middle ground: a checker exists on the training tasks, and a learned monitor must transfer
  to new tasks or constraint types. That is the out-of-domain probe and judge setting of 2611.09828, already
  studied, with evasion as the open problem.
- So the open problem is "monitors that stay robust under optimization pressure." That's a different research
  question from this proposal, and it's crowded.
