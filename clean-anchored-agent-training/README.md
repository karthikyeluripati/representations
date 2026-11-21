# clean-anchored-agent-training

This repository tests one hypothesis. Counterfactual-twin clean anchoring, added to outcome-verified corrective preference learning (CACRT), might improve tool-agent repair and transfer without destroying behaviour that was already correct.

**Status:** the CPU stage is complete and validated. **No model has been trained yet.** Training needs one authorized H100.

- `docs/`: read in this order:
  1. `research_thesis.md`
  2. `exact_prior_art.md`
  3. `method_specification.md`
  4. `training_protocol.md` (pre-registered rules)
  5. `compute_budget.md`
  6. `reproduction_protocol.md`
  7. `baseline_results.md`
  8. `experiment_log.md`
  9. `paper_plan.md`
- `src/`:
  - `environment/`: BFCL v4 episodes and faults
  - `replay/`: positional-correct replay and checkpoints
  - `counterfactuals/`: verified pairs, twins, and GPU policy sampling
  - `training/`: renderer, objectives, trainer, runner
  - `baselines/`: arms B0–B9 and EXP1–3
  - `evaluation/`: agent loop, suites, metrics
- `scripts/`:
  - `audit_sources.py`, `validate_environments.py`, `construct_correction_pairs.py`
  - `train_balanced_sft.py`, `train_localized.py`, `train_clean_anchored.py`
  - `evaluate.py`, `analyze_interference.py`, `measure_budget.py`
  - `run_pilot_h100.sh`
- `data/`: `tasks_v1.json` (sha256 15ac3b2b…) and `cacrt_pairs_v1/` (has a manifest)
- `configs/pilot.json`
- `results/`: `validation/`, `audit/`, `budget/`, `eval/*.summary.json`

**Setup:** `bfcl-eval==2026.3.23` (install with `--no-deps`), torch, transformers 4.57, peft 0.17.1. Run the tests with `python -m pytest tests`.
