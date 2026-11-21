# corrective-training-study

This is a subproject of `clean-anchored-agent-training`. The CACRT experiments are untouched, and CACRT is closed because its gate failed. The study asks whether established verified corrective training fixes π₀'s false-outage behaviour.

- **Baseline A:** balanced corrective SFT.
- **Baseline B:** CSO-style verified DPO, using TRL 0.24.

Start with:
1. `docs/published_baseline_audit.md`
2. `docs/protocol.md`
3. `docs/results.md` (π₀ failure taxonomy)
4. `docs/compute_budget.md`

**GPU run:** `bash scripts/run_baselines_h100.sh` (phases: setup, pi0, bench, dev, final). It needs authorization and the archive at `$WORK/pod_backup_20261119.tgz`.

**CPU tests:**
- `CTS_TOKENIZER=<Qwen3 tokenizer dir> python -m pytest tests`
- `python scripts/check_archive.py`
- `python scripts/build_cso_pairs.py --tokenizer <dir> --dry-run`
