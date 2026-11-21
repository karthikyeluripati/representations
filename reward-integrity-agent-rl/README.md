# reward-integrity-agent-rl

Do RL post-training agents learn to exploit incomplete verifiers in multi-step tool workflows?

**Status: STOP** (see `docs/research_decision.md`). No RL was run.

## Layout
- `src/environments/workflow_env.py`: deterministic sandboxed workflows (invoice, inventory), JSON tool interface.
- `src/verifiers/verifiers.py`: weak W1 (self-consistency) and W2 (+ spot check), the strong verifier, a single-step noise control, and FP-mechanism attribution.
- `src/agents/scripted.py`: legitimate solution variants, diagnostic error policies, and a stochastic imperfect worker. **Validation only.**
- `scripts/validate_environment.py`, `scripts/process_check_ablation.py`: CPU experiments. Outputs go to `results/`.
- `tests/test_env_verifiers.py`: 24 tests.
- `docs/`: novelty_audit, experimental_protocol, experiment_log, compute_budget, research_decision.

## Reproduce
```
python -m pytest -q tests
python scripts/validate_environment.py --n 200
python scripts/process_check_ablation.py --n 400
```
