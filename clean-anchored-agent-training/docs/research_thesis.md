# Research thesis

## Hypothesis under test
When an agent is trained on outcome-verified corrections, the harm to its existing behaviour is concentrated at the decision boundary the correction moves. For tool errors, that boundary separates three situations:
- a real error that needs a repair;
- an injected outage that needs a retry;
- a success that needs the agent to proceed.

So anchoring each correction's executed, verified **counterfactual twins** should preserve correct behaviour and improve transfer better than three alternatives:
- (a) no anchor;
- (b) generic clean replay or KL at the same anchor budget;
- (c) ordinary balanced SFT on the same states.

## Why this problem, in this setting
The previous project showed the following. Its outage-recovery SFT on BFCL v4 multi-turn (Qwen3-4B) learned the recovery rule, but applied outage recovery to 38% of real held-out API errors, against 5% for the untrained model. That training set contained 1,568 outage events and about 3 real-error repairs. These are observations from saved outputs, not a controlled result: **imbalance has not been shown to be the cause.**

The arms here are built to make that comparison controlled:
- B3 (balanced SFT from base) vs stage-0 π₀ (imbalanced) isolates imbalance.
- EXP1, EXP2 and EXP3 isolate the learning mechanism given the same corrective states.

## What would count as success (pre-registered, `training_protocol.md` §5)
EXP3 must clear the strongest matched baseline (EXP1, EXP2, B7_replay) on two fronts:
- **Held-out family (VehicleControlAPI):** genuine-error repair improves by ≥ 5 points.
- **Fault-free success:** it loses ≤ 2 points relative to π₀.

The improvement must also appear on real errors, not only on outages. In addition, EXP3 must beat B7_replay; otherwise the twin-selection claim fails even if anchoring helps.

## What this project is not
- **Not a new verification or localization method.** That part is CSO, and it is reported as CSO.
- **Not a new gradient-surgery or regularization method.** PCGrad, A-GEM and KL appear only as baselines and ablations.
- **Not a benchmark paper.** The benchmark is official BFCL v4 multi-turn execution and grading, with the positional-argument replay bug fixed, injected-outage overlays, and forced-prefix real-error episodes.

## Known threats
- **Single environment.** PAFT's OpenAgent cannot be reproduced (its POI database is not released and the rewriter needs a paid API), so the second-environment requirement is unmet. See `reproduction_protocol.md`.
- **Power.** The held-out family has 49 base tasks and 100 real-error states spread over 44 of them. Only effects of roughly 15–20 points or more on held-out repair can be detected reliably. The false-outage rate (hundreds of events) is more sensitive.
- **Error-type concentration.** Train real errors are dominated by TravelAPI argument and auth errors ("Card not registered", "Invalid access token"). Held-out errors are mainly Vehicle preconditions. This is a genuine shift in error category, but it is a single shift.
