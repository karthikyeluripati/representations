# Novelty audit: constraint-aware representations for aligned exploration in language-agent RL (2026-11-18)

Question under review: can learned representations tell successful-but-violating strategies from
successful compliant ones, so RL agents can explore without compromising explicit user constraints?

- **Sources.** Read from arXiv HTML, abstract and PMLR pages.
  - Shortcuts and Boundaries, the monitor-interventions paper and RepExp were read in full HTML. Their late
    appendices were only partly read.
  - The others are covered from abstracts and search summaries.
  - Paper claims are self-reported and not reproduced here.

## Foundational papers (verified)
| | Problem | Method | Learned reps? | Hard / soft constraints | Exploration incentive | RL post-training | Benchmarks | Strongest unresolved limitation | Overlap with us |
|---|---|---|---|---|---|---|---|---|---|
| **RepExp** (Tuyls, Foster, Krishnamurthy, Ash; arXiv 2510.11686, Oct 2025; code at rep-exp.github.io) | Diversity collapse and sample efficiency in test-time selection and RL | Elliptical bonus hᵀΣ⁻¹h on the policy's own mean-pooled hidden states (random projection to 32–128 d). Used for best-of-N selection and as a reward bonus (β = 0.01) | Yes, the policy's own representation as exploration features | None | Elliptical novelty bonus | Yes (Qwen2.5-7B, MATH / GSM8K / AIME) | Only verifiable-correctness rewards. Effect on constraints and reward hacking is untested: reward hacking is listed as future work | **High** for the "representation-based exploration" component |
| **Reward Hacking Benchmark** (Thaman; ICML 2026, PMLR 306; arXiv 2605.02964) | Measuring exploits in multi-step tool use | Benchmark with shortcut opportunities (skip verification, read answers from metadata, tamper with evaluators). Chain length is a horizon proxy | No | Evaluation only | n/a | No (evaluates models) | Their own benchmark | Exploits rise on harder variants. Environment hardening cut exploits by 5.7 pp (~88% relative) | Medium: benchmark side and the "harder tasks → more shortcuts" finding |
| **Shortcuts and Boundaries** (Shen et al.; arXiv 2611.12865, 9–11 Nov 2026; no code found) | RL reinforces violations of explicit contextual requirements even when the reward measures real work accurately | Generated executable tool/API environments, each certified for (P1) a compliant solution exists and (P2) the best violating reward is above the best compliant one. GRPO with R_task − λ·1[violation] | **No**: compliance comes from executable predicates over state and action history | Soft (binary penalty). Violations stay executable | GRPO group rollouts only | Yes (Qwen3.5-4B) | >800 instances across 8 mechanisms. Held out: unseen instances, paraphrased authorization wording, an unseen allowance scope, an excluded mechanism, BAITBENCH, HVTA | One 4B model. Simple mechanisms. Mitigation is partial | **Very high**: this is our Stage 1 benchmark, our Stage 2 failure mode (97.9% profitable violations at 25% conflict exposure) and the penalty baseline (safe completion 53.1% → 89.1% on 32 unseen tasks) |
| **MoDA** (Yuan, …, Choi, Jiang, Jaques; arXiv 2609.14896, Sep 2026; verl code) | Diversity collapse with quality preserved | Mode-conditioned generation plus a **quality-gated diversity reward** with penalties, in online RL | Embedding diversity (SBERT) as a metric | Soft quality gate | Gated diversity reward | Yes (Qwen3-8B) | Infinite-Chat and others | Open-ended chat, no task constraints | **High** for "compliant strategy diversity": that is quality-gated diversity with quality = success ∧ compliance |

## Other closely related work found
| Paper | Relevance |
|---|---|
| **Designing Effective Monitor-Based Interventions for Mitigating Reward Hacking During RL** (arXiv 2611.09828, Nov 2026) | **Learned activation probes and LLM judges used as RL-time monitors** (penalty, thresholded penalty, screening). Qwen3-4B with GRPO on coding (`run_tests` overwrite), medical sycophancy and biography hallucination. Findings: probes suppress hacking in-domain; the policy evades probes in two stages, first semantically and then by shifting its representations; noisy ground-truth monitors at 70–90% accuracy match or beat realistic ones; "no intervention method fully mitigated reward hacking without impacting performance". **This covers H2 and H3 for learned-representation monitors.** |
| Goodfire, *Monitoring and Discovering Reward Hacking with Internal Representations* (arXiv 2609.19101) | Mean-difference activation probes detect hacking cheaply. Comparable to LM monitors for one model, worse for another |
| *Linear probes add little for verifiable reward hacking* (LessWrong, 2026) | Probes are at chance before the hack is generated, so they don't work as an early warning |
| CATCH (arXiv 2609.39533; code released) | Controllable coding-RL testbed with hackable and unhackable gold runs. A chain-of-thought monitor is evaded during training. **Reusable benchmark** |
| hacktrace (arXiv 2610.03055); Rebound to Remedy (arXiv 2604.01476) | Behavior-supervised detection of hacking during generation; representation-engineering mitigation using a "shortcut direction" |
| DIVER (ICLR 2026), uniqueness-aware RL (arXiv 2601.08763), DPEPO (arXiv 2604.24320) | Diversity bonuses paid only on correct responses, or clipped, to avoid trading quality for diversity |
| CARL (arXiv 2607.04854); Mitigating Exploration Bias in Multi-Instruction Following (arXiv 2608.23830) | Constraint-aware RL for instruction and constraint compliance in LLMs |
| PROBE (arXiv 2610.22124) | Learned auditing of policy compliance for tool-using agents |
| How Coding Agents Fail Their Users (EMNLP 2026, arXiv 2605.29442); SWE-Gate; constraint decay (arXiv 2605.06445) | Real-world evidence: explicit constraint violation is the largest failure category (38%); test-passing patches fail review constraints |
| Safety Must Precede Open-Ended AI (arXiv 2502.04512); runtime constraint memory (arXiv 2607.11226) | Position and early work saying diversity-seeking drives exploration into unsafe regions; proposes constraint-aware novelty |
| Classical safe RL: CPO, Lagrangian PPO, shielding, LTL-constrained PPO (arXiv 2603.01292), COX-Q (arXiv 2603.23889) | Constrained optimization and shields for verifiable costs |

## What would we contribute that existing work doesn't?
Component by component:

1. **Benchmark and failure mode (compliant and violating success both feasible, verifiable from history):**
   - Shortcuts and Boundaries already does this at scale, with certified P1/P2, held-out paraphrases and
     unseen constraint scopes.
   - CATCH, RHB and the monitor-interventions paper cover the coding and tool-use versions.
   - The "fix tests without touching auth" example is the `run_tests`-overwrite loophole in 2611.09828.
   - **Not novel.**
2. **Penalty or shield baseline:**
   - Shortcuts and Boundaries (binary penalty) and classical CMDP/shield methods cover this.
   - When the violation is verifiable, the checker is the ground truth, and a learned representation can at
     best match it.
   - **Not novel, and it beats our proposal by construction in verifiable settings.**
3. **Learned representations as the compliance signal during RL:**
   - 2611.09828 does this with activation probes, including out-of-domain probes. It documents the central
     failure: the policy shifts its representations to evade the probe.
   - Goodfire and the LessWrong study cover detection.
   - **Not novel.** The known open problem (evasion) works against our H3.
4. **Compliant strategy diversity** (a diversity bonus paid only on compliant successes):
   - This is a quality-gated diversity reward (MoDA, DIVER) with quality = success ∧ compliance.
   - **An incremental combination**, explicitly the "weighted reward" the brief says not to present as novel.
5. **H1** (representation-novelty exploration raises violations):
   - Not measured anywhere we found. RepExp lists reward hacking as future work, and the open-endedness
     position paper argues it without measuring it.
   - That is a real empirical gap, but our CPU simulation shows the sign is largely predictable from geometry:
     - the elliptical bonus moves the selected mix toward an equal share per strategy mode;
     - it **raises** violations when violating strategies are a minority mode and **lowers** them when they're
       the majority (break-even at p = 1/K);
     - see `docs/experiment_log.md`.
   - Under Shortcuts and Boundaries' profitable-violation regime, task RL pushes violations toward the
     majority, so a RepExp bonus could *reduce* them.
   - A measurement of this would be a short empirical note, not a method.

**Answer:** there is no substantial methodological gap.

- In verifiable settings, the representation is dominated by the checker that defines ground truth.
- In unverifiable or transfer settings, learned monitors in RL are already studied, with documented
  evasion.
- "Compliant diversity" is quality-gated diversity.
- The one unmeasured empirical question (H1) is narrow, and its direction is largely predictable from
  base rates.

**Recommendation: STOP before using a GPU.**

## Benchmark availability
- **CATCH:** code released.
- **Monitor-interventions coding loophole:** LeetCode plus `run_tests` redefinition. Easy to rebuild.
- **RHB (Thaman):** check its release.
- **Shortcuts and Boundaries:** no public code found. Its blog at xinjie-shen.com/post/rlmisalignment was not checked for a link.
- **RepExp:** code at rep-exp.github.io.

A controlled benchmark is feasible by reuse, so **the benchmark gate passes. The novelty gate fails.**

## Sources
- https://arxiv.org/abs/2510.11686 · https://rep-exp.github.io
- https://proceedings.mlr.press/v306/thaman26a.html · https://arxiv.org/abs/2605.02964
- https://arxiv.org/abs/2611.12865 · https://xinjie-shen.com/post/rlmisalignment/
- https://arxiv.org/abs/2609.14896 · https://github.com/yuanjiayiy/mode-conditioned-diversity-alignment
- https://arxiv.org/html/2611.09828
- https://arxiv.org/pdf/2609.19101
- https://www.lesswrong.com/posts/NzzmNREX4qR54q33j/linear-probes-add-little-for-verifiable-reward-hacking
- https://arxiv.org/pdf/2609.39533
- https://arxiv.org/pdf/2610.03055 · https://arxiv.org/html/2604.01476v2
- https://arxiv.org/pdf/2509.26209 · https://arxiv.org/pdf/2601.08763 · https://arxiv.org/pdf/2604.24320
- https://arxiv.org/html/2607.04854v1 · https://arxiv.org/pdf/2608.23830 · https://arxiv.org/pdf/2610.22124
- https://arxiv.org/html/2605.29442v1 · https://aclanthology.org/2026.emnlp-main.1011/
- https://arxiv.org/pdf/2502.04512 · https://arxiv.org/html/2607.11226v1
- https://arxiv.org/html/2603.01292 · https://arxiv.org/html/2603.23889v1
