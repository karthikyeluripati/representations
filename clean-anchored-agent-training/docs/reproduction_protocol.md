# Reproduction protocol and environment status

## Environments
| environment | status | reason |
|---|---|---|
| **BFCL v4 multi-turn** (bfcl-eval 2026.3.23, Apache-2.0; gorilla @6ea5797) | **ready, primary** | official executable classes and state-based grading run locally on CPU and deterministically |
| **PAFT OpenAgent** (@769692c, MIT) | **blocked** | the POI databases `data/database_en.csv` and `data/database_test_en.csv` are not in the repo or on Hugging Face (only `examples/sample_database.csv`, 446 bytes); the LLM rewriter needs `OPENAI_API_KEY` (paid, not authorized); `eval/` needs vLLM on a GPU. Substituting a self-made POI table would amount to building a custom simulator, which the brief forbids, and the results would not be comparable to the paper's |

**Unblock options** (user decision):
- (a) ask the PAFT authors for the POI CSVs;
- (b) authorize a small paid-API budget to regenerate the rewritten queries;
- (c) accept a second BFCL-style environment instead: leave-one-family-out folds on TravelAPI and TradingBot. These are independent tool families but the same benchmark, so they are weaker than a second environment.

## The BFCL benchmark as used (`cacrt_tasks_v1`, sha256 15ac3b2b…)
- **Episodes.** One user turn per episode; earlier turns are executed ground truth shown as history. 400 turn-tasks were built, 334 BFCL turns were dropped (no state-changing call), and 2 base tasks are unassigned and unused.
- **Splits.** Family-grouped and fixed before training:

  | split | turn-tasks | base tasks |
  |---|---|---|
  | train | 187 | 89 |
  | dev | 54 | 29 |
  | test | 43 | 22 |
  | heldout | 114 | 49 |

  Held-out contains every base task that involves VehicleControlAPI, so all of its turns stay together. The other splits keep the previous project's v2 assignment by base task.
- **Fixed defects.**
  - **Positional-argument replay.** The fixed replay changes the end state of 7 held-out, 2 train and 1 unassigned task (`results/validation/environment_validation.json` V1). Every grader now uses signature-aware binding.
  - **Turn and call budget.** The cap is 20 tool calls and 20 generation steps (was 8 steps). The reference policy's worst case is 16 calls (S2), so no reference episode reaches the cap.

## Validation results (all CPU, reproducible with the scripts)
- **V2 solvability.** The executable reference solves S0/S1/S2 and correctly escalates S3 in 100% of episodes (seeds 0–2, all splits). No task is unsolvable.
- **V4 leakage.**
  - No user text is shared across splits, and no base task appears in two splits.
  - 8 ground-truth call *sequences* are shared across train/dev/test. These are BFCL template siblings, for example `cancel_order(order_id=12446)` in TradingBot tasks with different requests.
  - Held-out shares none with train; its only overlap, `retweet(tweet_id=10)`, is with test. In-distribution test scores are therefore optimistic for every arm. The held-out family is the primary readout.
- **V5 determinism.** 0 failures: a restored snapshot has an identical hash, and two independent replays give identical observations and state.
- **V6 error pools.** The V2, TRAIN and EVAL outage pools are disjoint, and none equals a real BFCL error. Training data contains no EVAL string and no `_replica` name (unit test).
- **End-to-end grader check** (`evaluate.py --mode scripted`, through the same text-parsing agent loop used for models):

  | policy | split | episodes | every suite | real-error repair | false-outage rate |
  |---|---|---|---|---|---|
  | oracle | heldout | 1,582 | 1.000 | 1.000 | 0.000 |
  | oracle | test | 619 | 1.000 | 1.000 | 0.000 |
  | outage-rule surrogate | heldout | 1,582 | 1.000 on S0–S3 and the shift suites | 0.000 | 1.000 |
  | outage-rule surrogate | test | 619 | 1.000 on S0–S3 and the shift suites | 0.000 | 1.000 |

  So the graders and the false-outage metric separate the two behaviours exactly.

## Running on a fresh machine
1. `pip install bfcl-eval==2026.3.23 --no-deps`, plus torch, transformers 4.57.x, peft 0.17.1 and (GPU only) vllm. Pins are in `results/audit/sources.json`.
2. `python scripts/validate_environments.py --v2-tasks <v2 data/tasks.json>`. Check that `tasks_sha256` equals `15ac3b2b3c1ef719a04a495905cc50a06a08ef1497561ebdc1c71a2b307dc056`.
3. `python scripts/construct_correction_pairs.py`. Check the file hashes against `data/cacrt_pairs_v1/MANIFEST.json`.
4. `python -m pytest tests`. 26 tests should pass.
5. `python scripts/evaluate.py --mode scripted --policy oracle` (and `--policy surrogate`). Expect the numbers in the table above.
