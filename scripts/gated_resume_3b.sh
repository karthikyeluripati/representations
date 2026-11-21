#!/usr/bin/env bash
# Resume the 3B dataset on this machine ONLY if its LM numerics are bitwise identical to the
# machine that produced the existing rows (otherwise one dataset would mix two policies).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
PY=.venv/bin/python
$PY scripts/numerics_fingerprint.py --config configs/baseline_pod.yaml --out /workspace/fp_3b_here.npz
$PY scripts/numerics_fingerprint.py --compare /workspace/fp_3b_h100a.npz /workspace/fp_3b_here.npz | tee /workspace/fp_compare.txt
if grep -q "IDENTICAL NUMERICS" /workspace/fp_compare.txt; then
  echo "numerics identical -> resuming 3B"
  exec bash scripts/run_shards.sh 8 results/decisions_v1_3b --config configs/baseline_pod.yaml \
       --rep-config configs/representation_pod.yaml --n-games 60 --n-rollouts 8
else
  echo "NUMERICS DIFFER -> NOT resuming 3B here (would mix policies). Human decision needed."
fi
