# Experiment log

## 2026-11-19: CPU stage (no GPU, no paid API)
- **Workspace.** C: had 188 MB free at the start and about 100 MB at the end; there is no other drive. A new isolated repository, `representations/clean-anchored-agent-training/`, was created with `git init` and no commits. The previous projects are untouched; only the v2 split map was read from `pspo_restored`.
- **Literature** (`exact_prior_art.md`):
  - Read in full: PAFT, Fission-GRPO, and Agent-STAR (arXiv HTML).
  - Read in part: Agent Error Dataset (Sec. 5.3 and App. A); CSO method section (from the PDF); RFT-forgetting Sec. 5.1.
  - DART: abstract only, because the PDF text did not extract.
  - Finding: CSO already contains Components A and B.
- **Repositories** were pinned in `results/audit/sources.json`. OpenAgent is MIT-licensed, but its POI data is not released, so it is blocked. Fission-GRPO, Agent-STAR and CSO have no licence.
- **Environment.**
  - Built `cacrt_tasks_v1` (400 turn-tasks).
  - Fixed positional-argument replay; this changes the end state of 7 held-out tasks.
  - Raised the call and step budget to 20.
  - Every validation check passes. 8 ground-truth sequences are shared within the trained families (documented).
- **Correction data.** 310 train corrective states, 262 of them with verified pairs (709 pairs), plus 524 verified twin anchors. Evaluation states: 100 held-out, 60 test.
  - Audit: the 9 RP states that failed the repair were skips of a context-setting call (for example `cd`), after which later calls had acted in the wrong directory. One forward repair step cannot fix these, and the verifier correctly excluded them.
- **Graders.** The scripted oracle scores 1.0 and the surrogate scores 0 on real-error repair with a false-outage rate of 1.0. Separation is exact.
- **Training stack.** Objectives, trainer (sum / PCGrad / A-GEM), LoRA path, and all arms. 26 unit tests pass, covering:
  - toy-gradient directions, masks, KL at the reference, and the gradient projections;
  - checkpoint and replay semantics, and no splicing;
  - that the anchored run on a toy problem keeps more probability on the protected action.
- **Real chat template.** 0 prefix-stability failures, max 5,997 tokens.
- **Not done.**
  - No model training or evaluation, and no measured GPU throughput.
  - No policy-sampled pairs (they need π₀).
  - No Fission-GRPO baseline.
  - The CSO and DART results sections were not read.

## 2026-11-19: H100 gate phase (pod session 07:28 to about 08:30 UTC)
- **Hardware and software.** 1× H100 80GB HBM3; driver 595.112.07; torch 2.8.0+cu129; vLLM 0.11.0; transformers 4.57.3; PEFT 0.17.1. Python 3.12 venv at /workspace/venv. `/workspace` is the 50 GB container overlay, not a separate persistent volume.
- **Data fixes made before any training.** Neither changes splits, task IDs or state counts.
  1. **Encoding.** On Windows, BFCL JSON had been read as cp1252, which garbled non-ASCII text in 134 task fields. It is now read as explicit UTF-8.
  2. **Hash-seed nondeterminism.** BFCL error strings print Python sets, so string-hash randomization changed their order between processes. `PYTHONHASHSEED=0` is now pinned, and output is written with LF line endings.
  - The canonical tasks file is now 1f912534…. Independent Windows and Linux builds are byte-identical.
- **Gate checks.** 27/27 unit tests pass; environment validation passes; pair-data hashes match.
- **STAGE0 → π₀.** 94 steps, 748 trajectories, 650 s.
  - Measured throughput is 0.8 s per sequence, below the 1.5 s threshold, so the 180-step pilot configuration stands.
  - Reference evaluations:

    | split | policy | fault-free | genuine repair | false-outage | S3 escalation |
    |---|---|---|---|---|---|
    | held-out | base | 0.439 | 0.36 | 0.009 | 0.023 |
    | held-out | π₀ | 0.649 | 0.88 | 0.355 | 0.991 |
    | test | base | 0.767 | 0.583 | 0.056 | 0.085 |
    | test | π₀ | 0.837 | 0.600 | 0.583 | 1.0 |

- **Policy verification** (K=4; 262 states):
  - π₀'s first action equals the repair in 626/1,048 samples; a+ branches succeed in 1,007/1,048 runs.
  - Accepted pairs: retry 123/262, escalate 252/262, alternative endpoint 144/185, π₀-sampled 48/54. Total: 567 pairs over 252 states.
  - Twins that π₀ already answers correctly: 515/524.
  - Cost: 7,689 model calls; 14,083 environment executions; 30.1 M input tokens.
- **Premise gate: FAILED.**

  | anchor set | mean cos(∇DPO, ∇anchor) at π₀ | sd |
  |---|---|---|
  | twin outage | +0.027 | 0.044 |
  | twin clean | +0.019 | 0.026 |
  | generic | +0.010 | 0.019 |

  Every anchor set is near-orthogonal to the DPO update, and the twins are no more conflicting than generic anchors. Following the brief, the arms were **not** run.
- **Backup.** `representations/pod_backups/pod_backup_20261119.tgz` (sha256 a460d0fa…, verified). It contains code, data, manifests, evaluation traces, logs and diagnostics. **Not backed up:** merged π₀ (7.6 GB), because there is no destination with enough space. It can be regenerated from the STAGE0 config in about 11 GPU-minutes.
