# Experiment log (append-only)

## 2026-11-19: CPU stage
- **Archive.** Located `representations/pod_backups/pod_backup_20261119.tgz`. Its hash, manifests and summary recomputation all pass. Recovered the 567 policy-verified pairs.
- **CSO audit.** Read the official repo @55e10b4: PRM-scored candidates; GOOD_MIN 0.6 / BAD_MAX 0.5; a single continuation of the chosen branch; LLaMA-Factory DPO (lr 5e-6, batch 4, 3 epochs).
- **Skill-CDPO.** Read the method section (criticality- and score-gap-weighted step DPO). It overlaps the brief's default candidate.
- **Failure taxonomy** of the archived π₀ traces:
  - false outage is the largest category, 162/403 held-out and 40/100 test;
  - it is not limited to unseen error types.
- **Implementation.** Built TRL-based Baseline A (balanced SFT, 189/189/189) and Baseline B (verified DPO, 567 pairs).
  - Full dry-run build: 0 split-tokenization failures.
  - TRL numerical tests: 3/3 pass.
  - Also implemented the recognition probe, the reproduction gate, and a resumable GPU launcher.
- **Not run:** any training or evaluation (no GPU authorized), the independent environment, and CSO-faithful mining of π₀'s own failed trajectories (see paper_readiness.md).
- **Disk.** C: has about 35 MB free.

## 2026-11-19: H100 baseline run (pod 103.207.149.137; 217 min of pod time)
- **Setup.** 27 + 3 tests pass; benchmark and archive hashes match.
- **π₀ rebuilt** (STAGE0, seed 0). Reproduction gate **failed** on test: episode agreement 0.966, but genuine repair +0.083 with the CI excluding 0. Held-out passed (agreement 0.931). Following the pre-registered rule, pairs were re-verified under the rebuilt π₀:
  - accepted retry 98/262, escalate 251/262, alternative endpoint 127/185, sampled 39/47;
  - 515 pairs over 251 states;
  - matched SFT set: 513 decisions = 171 × 3 categories.
- **Benchmark.** Projection 6.09 h > 5 h, so the pre-registered abort rule applied: one lr per arm (DPO 5e-6 = CSO default; SFT 1e-4). DPO ran at about 11.6 s/step and SFT at about 5.3 s/step, 387 steps each.
- **Results.** Single seed; final test and held-out evaluated once per frozen adapter. Full numbers are in results.md.
- **Backup.** `pod_backups/cts_slim_20261119.tgz` (sha256 d903e442…): adapters, traces, probe, logs, data and configs. Optimizer checkpoints and the merged π₀ were not backed up.

## 2026-11-19: attempt to preserve the exact π₀; C/D not run
- **π₀ on the H100 pod** b6ladq836ao563 (/workspace/models/pi0):
  - model-00001 sha256 07042b1f…, model-00002 sha256 8aee7004…;
  - tokenizer = Qwen3-4B-Instruct-2507 @cdbee75;
  - source = STAGE0 rebuild, seed 0.
- **Lossless compression options measured:**
  - bit-level patch vs base: 1.96 GB (49.9% of weights differ);
  - rank-32 factors + exact residual: about 990 MB.
  - Both are larger than the space free on local C:.
- **Plan approved by the user:** a RunPod network volume plus a CPU receiver pod.
  - AP-IN-1 had no CPU supply. That volume was created and then deleted, unused.
  - Created volume pdbasxf7ya (15 GB, EU-RO-1) and CPU pod msmqs12vqbvtpt.
- **Before the transfer started,** the H100 pod was stopped (API: "Exited by user", 13:41:14 UTC). It had no volume disk, so the exact π₀ is presumed lost. The receiver pod was terminated unused. The network volume pdbasxf7ya remains and is empty.
- **C (DPO + generic replay) and D (DPO + twin anchors) were not trained:**
  - the pre-registered gradient-conflict gate had failed;
  - the remaining authorized compute (about 1.8 h) was below the about 4 h a matched C/D pair needs;
  - no TRL-matched C/D implementation exists (the CACRT trainer differs from B in LoRA rank, lr, steps and pair set);
  - the exact π₀ is now unavailable.

## 2026-11-19: replication, seeds 0–2 (pod x30j1kjs0i2i4d, H100, US-NE-1, volume 9ghq6ztmfl)
- **Launch 1 failed:** the tar excluded `data/policy_pairs_v1.jsonl`, so the study tests errored. The watcher used `pgrep -f`, which matched its own ssh command and falsely reported progress. About 2.5 h idle. Fixed: file copied, pid file, `kill -0` watcher.
- **π₀:** rebuilt (648 s), SHA256SUMS + DONE on the volume. Pairs re-verified (twin_policy_correct 517/524). Tests passed: 27 + 4.
- **Bug: probe outputs overwrote evaluation traces.**
  - `run_replication_h100.sh` copies `results/probe/<name>.json*` into `eval/`. That glob matched the probe's `<name>.jsonl`, which replaced the 2,201-episode traces of A_s0 and B_s0 with 320 probe rows. Summaries were unaffected.
  - Fix: `evaluate_policies.py` now writes `<name>.probe_items.jsonl` / `<name>.probe_summary.json`, deployed mid-run (seeds 1–2 correct).
  - The first seed-0 re-evaluation was clobbered again by the stale old-name probe files. Those were deleted and seed 0 re-evaluated a second time. The superseded seed-0 summaries are kept in `replication/s0_rerun1/`.
- **Results:** see results.md, "Replication". Pre-registered "A > B" not met (B − A held-out +0.041 [0.009, 0.075], positive in 3/3 seeds). Intervention rule not met (1/3).
- **Backups:**
  - Full archive (393 MB, adapters included) on the volume: `/workspace/replication/replication_results.tgz`, sha256 1b7d3d38….
  - Slim archive locally: `pod_backups/replication_slim_20261119.tgz` (0.9 MB; traces, logs, pooled, configs), sha256 8aae5439…. The full archive is not stored locally because C: has 330 MB free.
- **Pod stopped** via the API after the backup. π₀, the adapters and the full archive persist on volume 9ghq6ztmfl (20 GB, about $1.40/month).
