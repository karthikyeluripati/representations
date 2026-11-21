# Experiment log (2026-11-18)

## Environment
| Item | State |
|---|---|
| Laptop | Windows 11, CPU only, C: about 55 MB free at start (project files only; no models or datasets) |
| H100 pod | From the evidence-repair project; still running (billing). **Not used**: GPU work is not authorized for this project |

## Literature
- Ten listed works were checked: eight verified on arXiv / PMLR, one found by exact title (2607.11022), and one inaccessible (preprints.org, 403).
- Two arXiv IDs returned transient 404s to the fetch tool. They were then confirmed through arXiv search and abstract pages.
- Outcome: see novelty_audit.md.

## Benchmark (`src/environments/workflow_env.py`, `src/verifiers/verifiers.py`)
- **Families:**
  - invoice: amount = qty × price × (1 − discount); ledger; report(total, count);
  - inventory: stock = opening + received − shipped; report(low-stock ids).
- **Verifiers:** W1 = report/ledger self-consistency; W2 = W1 + a one-record spot check; strong = full recomputation from source.
- **Single-step noise control:** response-independent flips.
- **Tests:** `tests/test_env_verifiers.py`, 24 pass. They cover:
  - deterministic resets;
  - 5 legitimate solution styles × 2 families pass every verifier;
  - each diagnostic FP is detected with its mechanism;
  - an inconsistent report fails everything;
  - the strong verifier equals an independent recomputation;
  - no tool exposes answers or verifier logic, and calc rejects code;
  - the noise control is response-independent.

## Validation (`scripts/validate_environment.py --n 200`, results/env_validation.json)
| Policy | Strong | W1 accept | W2 accept | FP share of W1-accepted | FP share of W2-accepted |
|---|---|---|---|---|---|
| Correct, all 5 styles, both families | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Diagnostic: fabricate all | 0 | 1.00 | 0.00–0.01 | 1.00 | – |
| Diagnostic: partial batch | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| Diagnostic: first only | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| Noisy worker p = 0.1, invoice / inventory | 0.33 / 0.31 | 0.96 / 0.97 | 0.72 / 0.80 | 0.66 / 0.68 | 0.55 / 0.61 |

**Mechanisms of FPs accepted by W1 (noisy worker, p = 0.1):**
| Family | Wrote unread record | Omitted records | Miscomputed |
|---|---|---|---|
| Invoice | 70 | 31 | 58 |
| Inventory | 68 | 37 | 62 |

## Process-check ablation (`scripts/process_check_ablation.py --n 400`, results/process_check_ablation.json)
| Family / slip rate | FP share of accepted, W1 | FP share of accepted, W1 + trace check | Residual mechanisms |
|---|---|---|---|
| Invoice 0.05 / 0.1 / 0.2 | 0.39 / 0.64 / 0.89 | 0.17 / 0.28 / 0.60 | 100% miscomputed |
| Inventory 0.05 / 0.1 / 0.2 | 0.38 / 0.67 / 0.86 | 0.16 / 0.36 / 0.52 | 100% miscomputed |

## Caveats
- All rollouts are scripted or stochastic synthetic workers. Their error model (independent slips) may not match LM behavior.
- In particular, LMs could produce cross-step binding errors (a value read for one record written to another). Those survive the trace check, and the synthetic worker does not generate them.
- That is precisely what the Stage A pilot would measure.

## Decision
STOP (see research_decision.md).
