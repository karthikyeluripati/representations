# Uncertain-Outcome Faults (UOF): release and reproduction guide
*For reviewers and users who want to rerun the benchmark or any result in the paper. The datasheet is in `paper/datasheet.md`.*

## Layout
| path | contents |
|---|---|
| `../src/environment/bfcl_env.py` | BFCL v4 episodes with injected faults, including the uncertain-outcome scenarios **ACa** (applied) / **ACl** (lost) and their wordings |
| `../src/environment/tau2_airline.py` | τ²-bench airline episodes (zero-shot second environment) |
| `../third_party/tau2_c06b07dd/` | vendored τ²-bench @c06b07dd (MIT, unmodified; see `README_VENDOR.md`) |
| `../scripts/feasibility_ambiguous_commit.py` | builds the BFCL diagnostic / decisive task sets by execution, with scripted validation |
| `../scripts/tau2_airline_feasibility.py`, `../scripts/tau2_validate.py` | the same for τ²-bench |
| `../scripts/eval_uncertain_outcome.py`, `../scripts/eval_tau2_ac.py` | evaluation of any vLLM model or LoRA adapter |
| `src/verification/outcome_twins.py`, `synthetic_apis.py` | execution-verified training-data builders |
| `scripts/build_*`, `scripts/render_*` | training sets for each study |
| `scripts/run_*_h100.sh` | one script per study (resumable; aborts on any checksum or validation mismatch) |
| `scripts/otv_analysis.py`, `pool_seeds.py`, `../scripts/tau2_analysis.py` | pre-registered analyses (paired task / event bootstrap) |
| `docs/` | every pre-registration, its edit log and its result, in chronological order |
| `paper/` | draft, datasheet, tables (`tables/leaderboard.*`) and figures (`figures/`) |

## Environment
- **Python:** 3.12.
- **Libraries:** `vllm==0.11.0`, `transformers==4.57.3`, `peft==0.17.1`, `accelerate==1.10.1`, `trl==0.24.0`, `bfcl-eval==2026.3.23` (install with `--no-deps`).
- **τ²-bench:** additionally `pydantic>=2.12`, `loguru`, `docstring-parser`, `deepdiff`, `addict`, `toml`, `python-dotenv`.
- **Environment variables:** `PYTHONHASHSEED=0`. BFCL error strings print Python sets, so this keeps runs identical.
- **Base model:** `Qwen/Qwen3-4B-Instruct-2507` @`cdbee75f17c01a7cc42f958dc650907174af0554`.
- **Hardware:** all GPU work ran on single H100 80 GB pods.

## Reproducing the benchmark (CPU only)
```bash
cd clean-anchored-agent-training
python scripts/feasibility_ambiguous_commit.py      # 116 diagnostic / 73 decisive tasks; scripted separation table
python scripts/label_idempotency.py                 # MCP-style annotations for 78 functions, by execution
python scripts/tau2_airline_feasibility.py third_party/tau2_c06b07dd results/tau2/airline_write_events.json
python scripts/tau2_validate.py                     # tau2 scripted separation table
python -m pytest -q tests && (cd corrective-training-study && python -m pytest -q tests)
```

## Reproducing the results (GPU)
| paper result | doc | run script |
|---|---|---|
| A vs B replication (standard faults) | `docs/protocol.md`, `docs/results.md` | `scripts/run_replication_h100.sh` |
| headroom over B | `docs/results.md` | `scripts/analyze_b_headroom.py` (CPU) |
| gap: prompting and scale | `docs/uncertain_outcome_study.md` | `../scripts/run_uncertain_outcome_h100.sh` |
| OTV, EGR, IAR, IAR + annotations | `docs/method_*.md` | `scripts/run_otv_h100.sh`, `run_egr_h100.sh`, `run_iar_h100.sh`, `run_iar_ann_h100.sh` |
| synthetic diversity (32 classes, 3 seeds) | `docs/method_synthetic.md` | `scripts/run_synth_h100.sh` (`SEEDS="0 1 2"`) |
| diversity scaling (64 classes) | `docs/method_synthetic.md` | `scripts/run_synth64_h100.sh` |
| τ²-bench zero-shot transfer | `docs/tau2_transfer.md` | `../scripts/run_tau2_eval_h100.sh` |
| check-all-writes | `docs/method_caw.md` | `scripts/run_caw_h100.sh` |
| semantic-counterfactual diagnostic | `docs/semantic_counterfactual_protocol.md`, `docs/semantic_cf_results.md` | `scripts/eval_semantic_cf.py`, `scripts/semantic_cf_analysis.py` (CPU validation: `scripts/semantic_cf_validate.py`) |
| Table 1, figures | — | `scripts/compile_leaderboard.py`, `scripts/make_figures.py` |

## Result archives (sha256)
Archives are in `../../pod_backups/`. Each holds per-episode traces, summaries, logs and run configs.

| archive | sha256 (prefix) |
|---|---|
| `replication_slim_20261119.tgz` | 8aae5439 |
| `ac_headroom_20261120.tgz` | 4e7809a4 |
| `uncertain_outcome_20261120.tgz` | a29ff5c6 |
| `otv_s0_slim_20261120.tgz` | a4dd26a1 |
| `egr_s0_slim_20261120.tgz` | 30958dbf |
| `iar_s0_slim_20261120.tgz` | c30648f4 |
| `iar_ann_s0_slim_20261120.tgz` | 6c5f7ccc |
| `synth_s0_slim_20261121.tgz` | 70656b05 |
| `synth_3seeds_slim_20261121.tgz` | b923b323 |
| `tau2_eval_slim_20261121.tgz` | 01ffa196 |
| `synth_analysis_scaling_20261121.json` | — |
| `semcf_eval_20261122.tgz` (semantic-counterfactual evaluation) | 2c6dc99b |

## Pre-registration practice
- **Each study** has a document that fixes its primary metric, matched baseline and decision rule **before training**.
- **Changes made before results** are appended to an edit log.
- **Every gate was applied as written.** Studies that failed their gate were stopped and are reported.

## Licences
- **New code and data:** Apache-2.0 (proposed).
- **BFCL-derived content:** Apache-2.0.
- **Vendored τ²-bench:** MIT (`third_party/tau2_c06b07dd/LICENSE`).
