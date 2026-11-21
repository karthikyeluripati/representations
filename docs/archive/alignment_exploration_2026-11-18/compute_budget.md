# Compute budget (2026-11-18)

## Local resources (inspected)
- Windows 11; 12 logical CPUs; 7.2 GB RAM.
- Python 3.12.10; torch 2.9.0+cpu. No CUDA, no `nvidia-smi`.

## Used in this session
- Literature audit: web only.
- `scripts/sim_elliptical_selection.py`: CPU, ~16 s.
- **No GPU, no pods.** All earlier RunPod pods are stopped.

## Estimate for the only candidate experiment (not authorized, not recommended)
H1 test-time pool, `docs/experimental_protocol.md`:
- **Generation:** 200 tasks × 64 samples × ~1.5k tokens ≈ 2 × 10⁷ generated tokens with Qwen2.5-7B, plus
  sandboxed test execution.
- **Throughput:** with vLLM on one H100 at roughly 2–4k tokens/s per sequence batch, about **2–4 GPU-hours**.
  Representation extraction adds about 0.5 h. CPU sandboxes would run in parallel.
- **Cost:** roughly **$6–15** on one H100 (rate assumed, not quoted).
- **Before any run:** benchmark 10 tasks × 8 samples first, to replace these assumptions with measured numbers.
- **What it would test:** only H1's sign relative to the base rate. A 7B model is the smallest size where
  RepExp reports benefits; it hurt Qwen-0.5B.
- **Cheaper alternatives:**
  - The sign is already predicted by the simulation.
  - A 0.5–1.5B model on CPU is too slow and outside RepExp's working regime.

## RL (Stage 5)
**Not estimated.** The gates failed. A matched 5-arm × 3-seed GRPO comparison at 4B would be tens of
H100-hours, and isn't justified.
