# Compute budget: "When Novelty Lies" (2026-11-18)

## Resources checked
- **Laptop:**
  - Windows 11, Python 3.12, torch 2.9.0+cpu, scikit-learn 1.7.2; no CUDA.
  - About 0.4 GB RAM free.
- **Disk:**
  - **C: had 0 MB free mid-session.** Writes failed with ENOSPC.
  - We recovered about 66 MB by deleting only our own temporary files: fetched PDFs, and duplicate extracted
    copies of pod-result tarballs in the session scratchpad. The tarballs themselves were kept.
  - The HF cache holds tokenizers only (no weights). Nothing large can be installed locally: no Chromium, no
    model weights.
- **GPU pod:** none used. The Brief 3 pod (`ifk83rgizoilvs`) remains stopped.

## What was run
| Run | Hardware | Time |
|---|---|---|
| Pair construction + 16 methods + bootstrap (seed 0) | laptop CPU | 26 s |
| Count-based exploration (4 families × 9 keys × 2 render conditions × 20 seeds) | laptop CPU | 21 s |
| Seed 1 replicate | laptop CPU | about 50 s |
| Tests (8) | laptop CPU | 1 s |

Total GPU-hours: **0**. Cost: **$0**.

## What a continuation would need (not authorized, not recommended now)
- **BrowserGym MiniWoB++ prevalence study** (the only step that could change the decision):
  - CPU-only: a Chromium container with about 4 vCPU and 10 GB disk;
  - about 2–4 h for about 50 tasks × perturbations, with scripted and random agents;
  - **no GPU.**
- **Learned history encoder** (only if prevalence is high and the probing baseline is costly):
  - a ≤ 1B text encoder fine-tuned contrastively on verified pairs;
  - about 2–4 GPU-hours on an L4 or A10 (about $2–4).
  - It resolves whether a recurrent LM learns which history events matter across apps (Gate D).
- **RL comparison:** not justified (see `research_decision.md`).

**GPU compute is not justified at this point.**
