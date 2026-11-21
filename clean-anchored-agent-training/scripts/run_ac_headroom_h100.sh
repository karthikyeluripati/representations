#!/usr/bin/env bash
# Ambiguous-commit headroom measurement (gate: corrective-training-study/docs/headroom_ambiguous_commit.md). Eval only.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJ=$(pwd); W=/workspace; R=$W/replication; PI0=$W/models/pi0; OUT=$W/ac_headroom
BASE=/root/models/Qwen3-4B-Instruct-2507; REV=cdbee75f17c01a7cc42f958dc650907174af0554
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false
mkdir -p /root/models /root/cache $OUT/logs
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" pytest hf_transfer
pip install -q --no-deps "bfcl-eval==2026.3.23"
(cd $PI0 && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: pi0 checksum mismatch"; exit 3; }
python -m pytest -q tests > $OUT/logs/tests.log 2>&1 || { tail -20 $OUT/logs/tests.log; exit 4; }
python scripts/feasibility_ambiguous_commit.py > $OUT/logs/feasibility.log 2>&1
grep -q "verify       ACa: 116/116  ACl: 116/116" $OUT/logs/feasibility.log || { echo "ABORT: scripted validation differs on pod"; exit 5; }
RUNS="pi0="; for S in 0 1 2; do RUNS="$RUNS B_s$S=$R/runs/B_s$S/adapter A_s$S=$R/runs/A_s$S/adapter"; done
python scripts/eval_ambiguous_commit.py --model $PI0 --runs $RUNS --out $OUT > $OUT/logs/eval_pi0_family.log 2>&1
[ -f $BASE/config.json ] || HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV', local_dir='$BASE')"
python scripts/eval_ambiguous_commit.py --model $BASE --runs base= --out $OUT > $OUT/logs/eval_base.log 2>&1
tar -czf $W/ac_headroom.tgz -C $W ac_headroom && sha256sum $W/ac_headroom.tgz
echo AC_DONE
