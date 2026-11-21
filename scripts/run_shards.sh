#!/usr/bin/env bash
# Run N decision-collection shards on one GPU, restarting any shard that crashes (e.g. CUDA OOM).
# Collection is resumable at the row level, so a restart continues where the shard stopped.
# Usage: bash scripts/run_shards.sh N OUT_DIR [extra collect_decisions args...]
set -u
N=$1; OUT=$2; shift 2
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Each process may hold at most ~1/N of the GPU (minus slack) so cached blocks can't starve the others.
export DARE_CUDA_MEM_FRACTION=$(python -c "print(round(0.97 / $N, 3))")
mkdir -p logs
for i in $(seq 0 $((N - 1))); do
  (
    for attempt in $(seq 1 20); do
      python -u scripts/collect_decisions.py --shard "$i/$N" --out "$OUT" "$@" >> "logs/$(basename "$OUT")_shard$i.log" 2>&1 && exit 0
      echo "shard $i attempt $attempt failed; restarting in 10s" >> "logs/$(basename "$OUT")_shard$i.log"
      sleep 10
    done
  ) &
done
wait
echo "all shards finished"
