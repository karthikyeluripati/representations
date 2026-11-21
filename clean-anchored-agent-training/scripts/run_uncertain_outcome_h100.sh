#!/usr/bin/env bash
# Uncertain-outcome study (pre-registered: corrective-training-study/docs/uncertain_outcome_study.md). Eval only; resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
W=/workspace; R=$W/replication; PI0=$W/models/pi0; OUT=$W/uncertain_outcome; M=/root/models
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $M /root/cache $OUT/logs
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" pytest hf_transfer
pip install -q --no-deps "bfcl-eval==2026.3.23"
(cd $PI0 && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: pi0 checksum mismatch"; exit 3; }
python -m pytest -q tests > $OUT/logs/tests.log 2>&1 || { tail -20 $OUT/logs/tests.log; exit 4; }
python scripts/feasibility_ambiguous_commit.py > $OUT/logs/feasibility.log 2>&1
grep -q "verify       ACa: 116/116  ACl: 116/116  S0: 116/116  ACa_para: 116/116  ACl_para: 116/116  ACa_implicit: 116/116  ACl_implicit: 116/116" \
    $OUT/logs/feasibility.log || { echo "ABORT: scripted validation differs on pod"; exit 5; }
dl() { [ -f $M/$1/config.json ] || python -c "from huggingface_hub import snapshot_download as s; s('Qwen/$1', revision='$2', local_dir='$M/$1')"; df -h /root | tail -1; }
ev() { python scripts/eval_uncertain_outcome.py --out $OUT "$@"; }

ev --model $PI0 --runs B_s0=$R/runs/B_s0/adapter pi0= > $OUT/logs/eval_pi0.log 2>&1
dl Qwen3-4B-Instruct-2507 cdbee75f17c01a7cc42f958dc650907174af0554
ev --model $M/Qwen3-4B-Instruct-2507 --runs base4b= > $OUT/logs/eval_base4b.log 2>&1; rm -rf $M/Qwen3-4B-Instruct-2507
dl Qwen3-30B-A3B-Instruct-2507-FP8 5a5a776300a41aaa681dd7ff0106608ef2bc90db
ev --model $M/Qwen3-30B-A3B-Instruct-2507-FP8 --runs q30b_a3b= > $OUT/logs/eval_q30b.log 2>&1; rm -rf $M/Qwen3-30B-A3B-Instruct-2507-FP8
dl Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
ev --model $M/Qwen3-8B --runs q8b= --no-thinking > $OUT/logs/eval_q8b.log 2>&1
grep -h '^[a-zA-Z0-9_]* {' $OUT/logs/eval_*.log
tar -czf $W/uncertain_outcome.tgz -C $W uncertain_outcome && sha256sum $W/uncertain_outcome.tgz
echo UO_DONE
