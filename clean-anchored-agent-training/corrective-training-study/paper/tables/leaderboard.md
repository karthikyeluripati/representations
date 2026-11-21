| policy | seeds | S0 | S1\|S0 | explicit | para | implicit | τ² check (implicit) |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 (base) | 1 | 0.810 | 0.968 | 0.489 | 0.495 | 0.495 | — |
| Qwen3-4B-Instruct-2507 (base) + check prompt | 1 | 0.828 | 0.979 | 0.500 | 0.505 | 0.495 | — |
| Qwen3-8B (no thinking) | 1 | 0.759 | 0.943 | 0.472 | 0.489 | 0.477 | — |
| Qwen3-8B (no thinking) + check prompt | 1 | 0.759 | 0.943 | 0.489 | 0.494 | 0.494 | — |
| Qwen3-30B-A3B-Instruct-2507 (FP8) | 1 | 0.888 | 1.000 | 0.500 | 0.500 | 0.500 | — |
| Qwen3-30B-A3B-Instruct-2507 (FP8) + check prompt | 1 | 0.897 | 1.000 | 0.534 | 0.649 | 0.500 | — |
| π₀ (corrective SFT) | 1 | 0.845 | 1.000 | 0.500 | 0.500 | 0.500 | — |
| π₀ (corrective SFT) + check prompt | 1 | 0.845 | 1.000 | 0.500 | 0.500 | 0.500 | — |
| B: CSO-style verified DPO | 1 | 0.888 | 1.000 | 0.510 | 0.510 | 0.510 | — |
| B: CSO-style verified DPO + check prompt | 1 | 0.888 | 0.981 | 0.505 | 0.505 | 0.505 | — |
| SYN-old (synthetic APIs, retry convention) | 3 | 0.862 | 0.987 | 0.656 | 0.674 | 0.631 | 0.000 |
| SYN-32 (32 synthetic classes, IAR) | 3 | 0.859 | 0.997 | 0.694 | 0.778 | 0.676 | 0.000 |
| SYN-64 (64 synthetic classes, IAR) | 3 | 0.856 | 1.000 | 0.729 | 0.805 | 0.696 | 0.026 |
| scripted verify-then-decide (reference) | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| scripted blind retry (reference) | - | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.000 |
