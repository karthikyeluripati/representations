# Method hypothesis (deliberately unwritten until the baselines are measured)

**Status:** no method is proposed. The brief requires Baselines A and B to be trained and their remaining failures measured first.

## What is known now
- π₀'s largest failure is the outage response to real errors: 162 of 403 failed held-out episodes, 40 of 100 failed test episodes.
- This happens for seen and unseen error types alike.
- Recognition vs choice is not yet known; the probe has been implemented and not yet run.

## Decision tree, fixed now and applied after the baseline results
1. **A or B removes false-outage failures** (held-out B-label episodes ≤ 0.3× π₀'s, CI excluding 0) without fault-free loss over 2 points. Then the ordinary baseline solves the measured problem. Report **BASELINE ALREADY SOLVES THE PROBLEM** and do not invent a method.
2. **B ≈ A** (CI of the difference contains 0). Preference optimization adds nothing over balanced data on this problem. Report it; any method must then beat A.
3. **A residual failure remains in both.** Classify it with the taxonomy and the probe:
   - **wrong probe letter** (no recognition): the deficit is in representation or evidence reading. Candidates are evidence-contrastive data (the same call returning a real error vs an outage); the closest prior work is PAFT EFP and contrast-set augmentation.
   - **right letter, wrong action** (choice deficit): the deficit is in action selection given the error type. Candidates are preference pairs whose negatives are the policy's *own* outage responses at real-error states, weighted by measured criticality. **This is Skill-CDPO's weighting** (Eq. 11), which must be a direct baseline. A distinct contribution would have to show what Skill-CDPO-style weighting leaves unresolved.
4. **Mandatory direct baselines for any proposed intervention:**
   - data balancing (A);
   - hard-example mining (pairs only where π₀ fails);
   - curriculum;
   - importance / criticality weighting (Skill-CDPO-style);
   - error-aware SFT;
   - contrastive augmentation.

   At most **one** intervention will be tried, and only if one of these leaves a measured gap.
