# Compute budget (estimate; the GPU benchmark step replaces it with measured numbers)

**Basis:**
- The only measured throughput on this pipeline is 0.8 s per ~4.3k-token sequence (forward + backward, LoRA, bf16, gradient checkpointing, 4B), from the STAGE0 run on the 2026-11-19 H100.
- TRL DPO/SFT throughput has **not** been measured yet. `PHASE=bench` runs 3 steps of each and projects the total.

| step | work | est. H100-h |
|---|---|---|
| setup (pip, 8 GB model, 27+3 tests, validation, archive check, dataset build) | — | 0.4 |
| rebuild π₀ (STAGE0, 94 steps) + reproduction evaluation (test + held-out, 2,201 episodes) | — | 0.35 |
| π₀ dev evaluation + recognition probe (dev, test, held-out) | — | 0.15 |
| benchmark (3 steps × 2 arms) | — | 0.05 |
| B: DPO × 2 lr (425 steps; about 8 policy sequences + reference forward per step, ≈ 8.5 s) | 2 × 1.0 | 2.0 |
| A: SFT × 2 lr (425 steps; 4 sequences per step, ≈ 3.2 s) | 2 × 0.4 | 0.8 |
| dev evaluations of 4 adapters (about 785 episodes each) | — | 0.25 |
| final test + held-out evaluation of 2 adapters + probe | — | 0.25 |
| **total** | | **≈ 4.3 (ceiling 5)** |

- **If pairs must be regenerated** (π₀ not reproduced): add about 0.2 h.
- **Abort rule:** if the measured projection exceeds 5 h, drop the second lr of each grid first.
- **Storage:** about 30 GB (base 8 GB, merged π₀ 8 GB, venv 3 GB, caches, four LoRA r8 adapters of about 70 MB each, evaluation traces of about 4 MB each).
- **Backup:** off the pod, about 30 MB without the models.
- **Pod billing:** this counts wall-clock pod time. Phases are resumable, so stop the pod between phases if needed.
- **Local disk:** C: has about 35 MB free. Nothing large may be written locally; the full DPO/SFT datasets (about 20 MB) are built on the pod only.

## Replication budget (`scripts/run_replication_h100.sh`; estimated from measured seed-0 throughput)
| step | H100-h |
|---|---|
| setup + tests + validation | 0.3 |
| π₀ rebuild (650 s) + π₀ evaluation | 0.3 |
| pair re-verification (K = 4) + dataset build | 0.3 |
| per seed: B (387 steps at ≈ 9–11.6 s; reference precomputed) | 1.0–1.25 |
| per seed: A (387 steps at ≈ 5.3 s) | 0.6 |
| per seed: 2 final evaluations + probe | 0.2 |
| **total, 3 seeds** | **≈ 6.3–7.0** |

- **Cost:** about $28–31 at $4.39/h. Hard ceiling: 7.5 H100-h.
- **Storage:** the network volume `pdbasxf7ya` (15 GB, EU-RO-1) holds only π₀ (8 GB) and the results (under 1 GB). The public base model, the venv and the caches live on the container disk (at least 40 GB) and are re-downloaded per pod.
