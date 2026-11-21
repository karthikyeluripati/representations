# Novelty audit: training-time verifier failure in multi-step tool agents (2026-11-18)

**Sources.** All arXiv items were verified on arXiv.
- **Read in full (HTML):** 2601.04411, 2610.35716, 2606.05263, 2610.15924, 2610.17824.
- **Abstract only:** 2609.35677, 2611.17287, 2607.11022.
- **Thaman (ICML 2026, PMLR 306):** abstract only; the PDF was unreadable.
- **Unverified:** "Strategic Verification for Long-Running LLM Agents" (preprints.org 202608.2057). The site returned 403 and nothing about it is used.

## Comparison matrix
| Paper | Exact claim | Real LLM RL? | Multi-step tools? | Response-dependent verifier errors? | Independent audits | Budgeted? | Exploitation shown in training? | Tracks FP change as policy changes? | Unseen workflows? | Code |
|---|---|---|---|---|---|---|---|---|---|---|
| Rate or Fate? RLVεR (2601.04411) | GRPO drift governed by Youden J; noise changes rate, not fate | Yes (Qwen2.5-3B, code, 5 seeds) | No | No (fixed i.i.d. FPR/FNR); time-varying named as future work | No | – | No | No (future work) | No | github.com/cognichip/Noisy-RL |
| Verifier Errors in RLVR (2609.35677) | Gradient-flow conditions for reward ↑ while correctness ↓; RLVR observations can't identify accepted errors; audit-based correction gives "selective control" | Bandits + "a language model" (abstract) | Not stated | Yes (fixed imperfect verifier) | Yes (audits) | Partial auditing | Characterized theoretically | Not stated | Not stated | Not stated |
| Audit Where Hacks Pay (2610.35716) | Low-pass budgeted group audits (LB) with reflected penalty suppress hacking; sign of frozen hack mass predicts suppression (33/39 runs) | Yes (Qwen3-4B LoRA; Qwen2.5-3B) | No (single-response code) | Yes, exploitable (test rewriting, own `run_tests`) | Strong verifier on weak passes (assumed reliable and independent of policy) | Yes (p ≈ 0.28–0.33 of weak passes) | **Yes** | Hack rates at checkpoints; no verifier drift | Held-out problems only | Not stated |
| NOPO (2611.17287) | One-sided verifiability: treat positives as unlabeled, update only on trusted negatives | Yes (code, logic, math) | No | Yes (partial checks reward shortcuts) | No (no oracle) | – | Motivation | Not stated | Cross-benchmark transfer | Not stated |
| When the Reward Suite Is Leaky (2607.11022) | Natural MBPP test-suite FPs are per-task, persistent, asymmetric. **FP incidence does not grow in training: selection of pre-existing errors, not learned exploitation** | Yes (preregistered 2-arm GRPO) | No | Yes (natural) | Human-adjudicated, every rewarded FP | Static leakiness audit | **No growth observed** | Yes (FP mass across training) | Two replication families | github.com/toffee-desuwa/rlvr-leaky-suite |
| CVT-RL / PCCC (2606.05263) | Constrained PG with counterfactual credit reduces hacking 7.2 → 3.9% in long-horizon agents | Yes (Qwen2.5-7B etc., 3,000 updates) | **Yes** (≤ 16 tool calls; ALFWorld, WebShop, WebArena, …) | Robustness at 5–20% noise; type not stated | Human audit, 600 trajectories | Not budgeted | Hacking measured on trained policies | No | Unseen task families (probe) | Not stated |
| Weak-to-Strong Verifier Hardening (2610.15924) | Hacker–fixer loop hardens agent-environment verifiers | **No RL** (named as future work) | Yes (Terminal-Bench, SETA, KernelBench) | Yes (successive bypasses) | Held-out frontier attackers | Attack-call budgets | Not in training | ASR by hardening iteration | Per-task only | Not yet released |
| Reward Hacking Benchmark (Thaman, ICML 2026) | Tool-use exploit benchmark; RL post-training associated with more hacking (0.6% vs 13.9%) | Not stated (observational across models) | Yes (independent and chained regimes) | Yes (shortcut opportunities) | Not stated | – | Observational | No | Not stated | Not stated |
| Checking the Checkers (2610.17824) | Survey of 58 studies: "structure of an error matters more than its rate" | Survey | Mostly single-response | Taxonomy: random / systematic / exploitable / policy-shifting | Cited | – | Cited (e.g. hacking within 100 steps) | Recommends FPR/FNR at the start and end of training | – | Unverified repo |

Also found: LEGO-RL (2608.17393), sandbox defenses against reward hacking for coding agents; and 2607.05773, internal-state validation in agent RL gyms.

## Critical novelty question
*Does the literature already establish and solve training-time verification failure caused by correlated, multi-step tool behavior under limited independent auditing?*

- **Established:**
  - H1 for single-response RLVR: Audit Where Hacks Pay shows exploitation during GRPO.
  - Leaky shows a counter-case: natural FPs that are selected, not learned.
  - Budgeted audits (LB), audit-based selective control, and negative-only learning (NOPO) are strong, published defenses.
  - Agent-level hacking is measured (CVT-RL, Thaman) and verifiers are hardened offline (2610.15924).
- **Not established:**
  - Budgeted-audit RL in **multi-step tool agents**, where false positives arise from **combinations of actions** (skipped reads, omitted records) rather than from the final output.
  - How those FP types change on-policy (H3 is named as future work by 2601.04411 and recommended by 2610.17824).
- **Our CPU evidence undercuts the gap's significance** (see experiment_log.md):
  - In an executable workflow benchmark, a **cheap deterministic trace check** (every listed record read and written) removes *all* composition-type false positives.
  - What remains is single-step miscomputation, which is exactly the setting of LB / selective control / NOPO.
  - The multi-step-specific part therefore matches the brief's stop rule: "a cheap deterministic correction eliminates the gap".

**Gate A (method novelty): FAIL** for "audit allocation for multi-step composition errors". The only defensible remainder (H3: on-policy FP drift measurement) is an empirical question, not a method, and it requires RL runs.
