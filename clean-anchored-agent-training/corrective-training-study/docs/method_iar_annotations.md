# Method study 4: IAR with MCP-style idempotency annotations (IAR-A). Pre-registered 2026-11-20, before training
**Motivation** (method_iar.md):
- IAR teaches "plain timeout on a write → check". Checking after the implicit timeout reaches about 96% within the trained API classes, but **0/76 on the held-out class**.
- So the learned rule is keyed to API identity, because nothing in an unseen API says which calls are unsafe to repeat.
- **Real tool protocols expose exactly this.** The Model Context Protocol defines tool annotations such as `readOnlyHint`, `idempotentHint` and `destructiveHint`.
- **Question:** if every tool description carries such annotations in training and in testing, does the learned rule follow the annotation to an unseen API class?

## Annotations
- **Derived by execution** (`scripts/label_idempotency.py`): every ground-truth call in every usable task is executed once and twice from the same state, using the grader's state hash.
  - read-only = the state never changes;
  - idempotent = twice equals once (conservative: any non-idempotent observation wins).
- **Result: 78 functions** labelled:
  - 36 read-only;
  - 28 idempotent writes;
  - 14 non-idempotent writes, including every decisive fault target (book_flight, send_message, comment, create_ticket, fund_account, withdraw_funds, register_credit_card, purchase_insurance).
- **Unobserved functions** get no annotation.
- **Rendering:** appended to the tool description as ` [annotations: readOnlyHint=…, idempotentHint=…]`, enabled with `CTS_TOOL_ANNOTATIONS` for **all** rendering and evaluation in this study. The text is neutral: it gives no instruction about what to do.
- **Labels describe the API, not the episode.** They are computed for all classes, the held-out one included, the way an API publisher would document them. No evaluation episode or outcome is used.

## Arms (fold T; identical data to method_iar.md, only the tool text differs; SFT from π_B, lr 1e-4, 3 epochs, seed 0)
| run | trained with annotations | evaluated with annotations |
|---|---|---|
| **IARA** (method) | yes (IAR data) | yes |
| **AAC3A** (matched baseline) | yes (old-convention data) | yes |
| IARTx (control: annotations only at test) | no (the existing IART_s0 adapter) | yes |
| piBA (control: annotations, no AC training) | — | yes |

The B anchors are pre-rendered text from B's data, without annotations, and are identical in both arms.

## Rules (fixed now)
- **Primary:** implicit-timeout success on held-out TravelAPI (the same 35 π_B-solvable decisive tasks as method_iar.md), IARA − AAC3A.
- **Gate for seeds 1–2.** All must hold:
  - ≥ **+0.15**;
  - para not lower than AAC3A by more than 0.05;
  - S1|S0 ≥ 0.95.
- **Secondary:**
  - checks after the implicit timeout on Travel (the 0/76 baseline);
  - IARA vs IARTx (does training *with* annotations matter?);
  - piBA (do annotations alone change behaviour?);
  - over-checking: extra calls on S0/S1;
  - the original test and held-out.

**Budget:** 2 SFT runs + 4 evaluations. About 1.5 H100-h, about $7.

## Seed-0 result (2026-11-20; `pod_backups/iar_ann_s0_slim_20261120.tgz`, sha256 6c5f7ccc…)
| | implicit, held-out Travel (primary) | para, held-out Travel | explicit, held-out Travel | implicit, in-distribution |
|---|---|---|---|---|
| IARA (IAR + annotations) | 0.514 | 0.671 | 0.571 | **0.917** |
| AAC3A (old convention + annotations) | 0.486 | 0.643 | 0.857 | 0.500 |
| IARTx (IAR, annotations only at test) | 0.471 | 0.614 | 0.486 | 0.750 |
| piBA (annotations only) | 0.486 | 0.514 | 0.486 | 0.500 |

- **IARA − AAC3A = +0.029 [0.000, 0.071]. The gate (≥ +0.15) failed.** Not continued.
- **Checks after the implicit timeout on held-out Travel:**
  - IARA 2/76;
  - every other run 0/76.

  **Annotations did not let the rule transfer.**
- **Annotations alone do nothing** (piBA = π_B).
- **In-distribution the implicit rule is strong:** 0.917 with annotations vs 0.750 without.
- **Costs:**
  - explicit wording on held-out Travel drops (0.571 vs 0.857);
  - original test 0.802 vs 0.867 (AAC3A) and 0.872 (B_s0);
  - held-out 0.712 vs 0.750;
  - S1|S0 0.99.
- **Reading:** a 4B model fine-tuned on 4 API classes does not learn "non-idempotent per annotation → check" as an abstract rule. It learns associations tied to the functions it trained on, even when the abstract property is printed in every tool description.
