# Research decision: "When Novelty Lies" (2026-11-18)

## Gates
| Gate | Result | Evidence |
|---|---|---|
| A: novelty | **Analysis: narrow PASS. Method: FAIL** | A history-conditioned learned behavioral metric is recurrent predictive bisimulation (TEB/DBC lineage) applied to text. APSG already addresses look-alike-but-different web pages with interaction evidence and no training. ScreenSearch flags ambiguous screens for probing |
| B: benchmark validity | **PASS (proxy only)** | Both failure modes, verified by exhaustive H = 3 execution in the WebState simulator, with identifiability labels. Not reproduced on BrowserGym (local disk full) |
| C: baseline insufficiency | **PARTIAL** | Text and history keys fail (B_det 1.00 or route 0.45–1.00). Canonicalization + no-commit probing reaches 0.07 balanced error on identifiable pairs at about 8 environment steps per comparison. Failures left: rename (action matching) and committal-only distinctions |
| D: learnability | **NOT MET** | About 50% of hidden-state false equivalences are unidentifiable from history. On the identifiable half, the best history-only rule has 0.28 balanced error. A learned combiner did not generalize (0.50). Perfect keys add ≤ 7% explored states |
| E, F | not run | Blocked by D; would need a GPU and RL |

## Answers
1. **Exact new contribution:**
   - A measurement framework scoring exploration rewards jointly on false novelty and false equivalence,
     with execution-verified H-step labels.
   - An explicit split of errors into history-determined vs history-ambiguous.
   - No new method survived.
2. **Why it isn't covered by predictive bisimulation or GUI exploration work:**
   - **The analysis isn't covered:** TEB assumes Markov states; APSG and ScreenSearch don't report
     identifiability; PSSE and DPEPO use exact or undefined equality.
   - **The method would be covered:**
     - a learned history-conditioned metric is predictive bisimulation with a recurrent encoder;
     - the false-equivalence half is what APSG and ScreenSearch already handle with interaction evidence.
3. **Are both failure modes reproducible?**
   - Yes, in our executable proxy:
     - cosmetic, rename and route false novelty;
     - hidden-state, delayed-effect and permission false equivalence.
   - Not yet on BrowserGym.
4. **Are the examples behaviorally verified?**
   - Yes. Exhaustive execution of every action sequence of length ≤ 3, with witness sequences and minimal
     divergence horizons (1–3).
   - 1-step equivalence is reported separately: 47% of test B_determined pairs are 1-step-equivalent but
     differ within 3 steps.
5. **Which simple baseline fails, and why?**
   - **Canonical-observation keys:** every hidden-state false equivalence (1.00), because the current
     observations are identical by construction.
   - **History keys (PSSE-style set, full history, tracker):** they flag equivalent routes as novel
     (0.45–1.00), because they can't tell which past events matter.
   - **All label-keyed methods, including probes:** they fail on renamed labels (1.00).
   - **No-commit probing** almost works on identifiable pairs (0.07), but costs interaction and needs
     reversible probes.
6. **Is the remaining problem learnable from observable history?**
   - Only for about half the false-equivalence cases.
   - On that half, the gap over simple rules is modest (0.28 → 0) and is already mostly closed by cheap
     probing.
   - The other half is unidentifiable without acting.
   - We found no evidence a learned model generalizes across apps; only a weak learner was tried.
7. **What method would address the limitation?**
   - An amortized probe: a history encoder trained to predict the outcomes of reversible probes, which saves
     the probe interactions. That is, a learned belief/predictive-state model.
   - It is conceptually standard, and the proxy suggests a small payoff.
8. **Is GPU compute justified?** No.
   - The one experiment that could change the decision, prevalence of history-determined aliasing in
     BrowserGym/MiniWoB++, needs only a CPU container with Chromium.
9. **Realistic path to a main-track paper:**
   - Not via this method. A benchmark/analysis paper is possible only if real web environments show that:
     - history-determined aliasing is frequent;
     - probing is costly or unsafe there;
     - exploration outcomes change materially.
   - The proxy gives no reason to expect all three.
10. **Decision:** STOP (below).

## Decision: **STOP**
- STOP the proposed learned history-conditioned novelty representation and RL program. The gates fail as
  follows:
  - **Gate A for the method:** it is not distinguishable from recurrent predictive bisimulation plus
    APSG-style interaction evidence;
  - **Gate C:** only partial; deterministic canonicalization plus reversible probing leaves little;
  - **Gate D:** not met; half the gap is unidentifiable, the other half small, and behavioral headroom is
    ≤ 7%.
- **Preserved:** the simulator, the identifiability-labelled benchmark, and the results
  (`results/strong_paper/`).
- **Optional, not recommended for a main track:** a CPU-only BrowserGym prevalence check. It is the only
  evidence that could reopen the question (kill criterion: history-determined aliasing in < 10% of
  explored state pairs, or no-commit probing below 0.1 balanced error).
