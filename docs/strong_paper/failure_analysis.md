# Failure analysis: is a learned history-conditioned representation necessary? (2026-11-18)

All numbers come from the WebState proxy (see `benchmark_protocol.md`).

## Reading these numbers
- **Splits:** held-out test families (mail, ticket), seed 0, with task-clustered 95% CIs. Seed 1 matches to
  within about 0.05 on every headline number (`results/strong_paper/seed1/`).
- **Metric:** error = false-novelty rate on equivalent pairs, false-equivalence rate on non-equivalent pairs.
- **What the proxy can and can't show:**
  - The environment was designed by us, so the *existence* of both failure modes is by construction.
  - What it can show is which kinds of method can or cannot fix them, and how much information the
    observable history carries.
  - It **cannot** show how often these situations occur in real web tasks.

## Both failure modes are reproduced and verified by execution

| Subtype (test n) | Exact obs | Canonical obs | Role-canonical obs | Transition (canonical) | History set (PSSE-style) | Field tracker | Mutation log | Full history | TF-IDF obs | Probe k = 2, no-commit | Probe k = 2 (snapshots) | Learned combiner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_render layout / decor / style (240 each) | 0.76 / 1.00 / 1.00 | **0** | **0** | 0 | 0 | 0 | 0 | 0 | 0–0.05 | 0 | 0 | 0 |
| A_render **rename** (held-out operator, 240) | 1.00 | 1.00 | **0.19** [0.14, 0.24] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.00 |
| A_route_determined (66) | 0 | **0** | 0 | 0.08 | 0.77 [0.67, 0.86] | 0.45 [0.28, 0.60] | 0.42 [0.12, 0.74] | 1.00 | 0 | **0** | 0 | 1.00 |
| A_route_ambiguous (94) | 0 | 0 | 0 | 0.34 | 0.72 | 0.53 | 0.64 | 1.00 | 0 | 0 | 0 | 1.00 |
| B_determined (129) | 1.00 | **1.00** | 1.00 | 0.53 | 0.05 [0.00, 0.10] | 0.37 [0.18, 0.54] | 0.14 [0.05, 0.27] | **0.00** | 1.00 | **0.14** [0.02, 0.30] | 0.00 | 0.00 |
| B_ambiguous (126) | 1.00 | 1.00 | 1.00 | 0.73 | 0.29 | 0.52 | 0.50 | 0.10 | 1.00 | 0.42 | 0.06 | 0.10 |

Balanced error on **history-determined** pairs, (A_route_determined FN + B_determined FE) / 2:

| Method | Balanced error | Cost |
|---|---|---|
| canonical obs | 0.50 | |
| full history | 0.50 | |
| history set | 0.41 | |
| field tracker | 0.41 | |
| mutation log | 0.28 | |
| learned combiner | 0.50 | |
| **no-commit probe** | **0.07** | about 8 extra environment steps per comparison |
| snapshot probe | 0.00 | about 14 steps, including committal actions on snapshots |

## Answers to Stage 3

1. **Does deterministic canonicalization fail?**
   - It removes cosmetic false novelty completely (layout, decor, style: 0 errors).
   - It fails on two things:
     - **Renamed labels.** Every label-keyed method has error 1.00. Role-only canonicalization gets 0.19,
       but then it can't separate same-role elements.
     - **Every hidden-state false equivalence:** B_determined 1.00, because the current observation is
       identical by construction.
2. **Does exact observation-transition matching fail?**
   - Yes, both ways. It has false novelty on routes (0.08–0.34) and still misses about half the false
     equivalences (0.53 / 0.73).
3. **Do existing behavior-aware abstractions fail?**
   - History-based keys fix false equivalence: PSSE-style history set 0.05, full history 0.00.
   - They create false novelty on equivalent routes: 0.45–1.00.
   - APSG-style **interaction evidence** (probing reversible actions and comparing outcomes) nearly fixes
     both on identifiable pairs (0.00 / 0.14), at an interaction cost. It fails on rename, because actions
     are matched by label, a limitation APSG itself acknowledges.
   - TEB was **not reproduced**: there is no code ("will be released later"), it is visual RL, and it
     assumes an MDP. We did not substitute a weaker method under its name.
4. **Does partial observability really prevent reliable novelty estimation?**
   - Partly. Among test pairs with identical current observations but different behavior, **49%**
     (126/255; seed 1: 51%) are *ambiguous*: the observable history does not determine the label.
   - The same holds for 59% of equivalent-route pairs (94/160).
   - No history-conditioned representation can be correct on these across completions.
   - Only acting can resolve them: snapshot probing reaches 0.06 on B_ambiguous.
5. **Can observable interaction history resolve the ambiguity?**
   - For the determined half, yes, in principle (Bayes error 0).
   - The deterministic history keys that exploit it pay with false novelty.
   - Of the history-only methods tested, the best is the mutation log at 0.28 balanced error. That is the
     **only learnable gap** we found.
6. **Would longer histories or ordinary recurrent models solve it?**
   - Not tested. No LM or GPU is available locally.
   - The naive learned combiner (logistic regression over deterministic key-agreement features, trained on
     settings/shop) **degenerates into the full-history key** on held-out families:
     - route false novelty 1.00, B_determined 0.00;
     - test AUROC 0.94 but balanced error 0.50.
   - So the gap is not learnable from these hand features. Whether a recurrent LM encoder learns *which*
     history events matter, and generalizes across apps, is the open question.
7. **What is fundamentally unidentifiable?**
   - Hidden variables never reflected in the history so far: role, membership, stock, offline, delay.
   - Within the proxy, about half of the false-equivalence cases are of this kind.
   - Resolving them requires interaction (probing). That is an exploration-policy question, not a
     representation question.

## Do the reward errors change behavior?
Count-based exploration (30 episodes × 8 steps, 20 seeds, random cosmetic renders) shows little headroom:

| Method | Unique true states (settings / shop / mail / ticket) |
|---|---|
| random | 41 / 57 / 31 / 47 |
| canonical-observation key | 45 / 61 / 32 / 49 |
| history keys (history set, tracker, full history) | within about 2 states of canonical; full history ≈ random |
| privileged **oracle state key** | 46 / 62 / 34 / 53 |

- Standard errors are about 1 state.
- The oracle beats the best non-privileged key by 0–3.4 states (0–7%).
- Success-discovery rates show no consistent ordering.
- Even with perfect reward reliability, exploration in this proxy improves only marginally.

## Verdict
- **Gate B (benchmark validity):** PASS in the proxy. Both modes are demonstrated and verified by
  execution, with H-step labels and identifiability labels. Not yet demonstrated on BrowserGym.
- **Gate C (baseline insufficiency):** PARTIAL.
  - Text and history keys fail.
  - Cheap interaction probing plus canonicalization leaves small errors on identifiable pairs (0.07
    balanced). The remaining failures are rename, which better action matching fixes, and committal-only
    distinctions.
- **Gate D (learnability):** NOT MET.
  - The learnable gap is limited to the history-determined half, where the best history-only rule has 0.28
    balanced error.
  - The one learned model tried did not generalize.
  - Behavioral headroom from perfect keys is small (≤ 7% more states).
- **Gate E / F:** not run, because Gate D failed and they need a GPU.
