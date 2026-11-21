#!/usr/bin/env bash
# One-time setup on a Linux GPU pod (e.g. RunPod PyTorch template, Ubuntu 22.04, CUDA 12.x).
# Persists caches on the network volume if /workspace exists.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d /workspace ]; then
  export HF_HOME=/workspace/hf_cache
  export ALFWORLD_DATA=/workspace/alfworld
else
  export ALFWORLD_DATA="$HOME/.cache/alfworld"
fi

python3 -m venv --system-site-packages .venv   # reuse the image's CUDA torch
source .venv/bin/activate
pip install -q --upgrade pip
# Use the pod's CUDA torch wheel (default PyPI wheel ships CUDA 12.x on Linux).
python -c "import torch" 2>/dev/null || pip install -q torch
pip install -q "transformers>=4.45,<5" accelerate alfworld==0.4.2 pyyaml pandas pyarrow scipy scikit-learn pytest
pip install -q -e .

alfworld-download --data-dir "$ALFWORLD_DATA"
# Point configs at the data location (default config assumes ~/.cache/alfworld).
if [ "$ALFWORLD_DATA" != "$HOME/.cache/alfworld" ]; then
  mkdir -p "$HOME/.cache"
  # A leftover real directory (e.g. from an interrupted run) would make ln put the link *inside* it.
  if [ -d "$HOME/.cache/alfworld" ] && [ ! -L "$HOME/.cache/alfworld" ]; then
    rm -f "$HOME/.cache/alfworld/alfworld"; rmdir "$HOME/.cache/alfworld"  # fails loudly if it holds real data
  fi
  ln -sfnT "$ALFWORLD_DATA" "$HOME/.cache/alfworld"
fi

python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
python scripts/check_environment.py --episodes 5
python -m pytest -q
DARE_LM_TESTS=1 python -m pytest -q tests/test_lm_scorer.py
echo "Setup OK. Remember: export HF_HOME=${HF_HOME:-~/.cache/huggingface}"
