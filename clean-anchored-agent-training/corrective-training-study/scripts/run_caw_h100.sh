#!/usr/bin/env bash
# Check-all-writes study (pre-registered: docs/method_caw.md). SEEDS per pod; pods share the volume.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; R=$W/replication; PIB=$W/models/piB; O=$W/synth; T2=$W/tau2_eval
SEEDS=${SEEDS:-"0"}; TAG=caw_$(echo $SEEDS | tr " " "_")
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CTS_TOOL_ANNOTATIONS
mkdir -p /root/cache $O/runs $O/uo $O/eval $O/logs $O/data $T2/logs logs $PROJ/logs
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" "trl==0.24.0" pytest hf_transfer
pip install -q --no-deps "bfcl-eval==2026.3.23"
pip install -q loguru docstring-parser deepdiff addict toml python-dotenv
(cd $PIB && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: piB checksum mismatch"; exit 3; }
(cd $PROJ && python -m pytest -q tests) > $O/logs/tests_parent_$TAG.log 2>&1 || { tail -20 $O/logs/tests_parent_$TAG.log; exit 4; }
python -m pytest -q tests/test_caw_data.py > $O/logs/tests_caw_$TAG.log 2>&1 || { tail -20 $O/logs/tests_caw_$TAG.log; exit 4; }
python scripts/check_manifest.py $STUDY/data/caw_v1/MANIFEST.json n=1349 n_synthetic=832
if mkdir $O/data/rendercaw.lock 2>/dev/null; then
  if [ ! -f $O/data/MANIFEST_rendercaw.json ]; then
    python scripts/render_sft_records.py --tokenizer $PIB --b-dpo $R/data/cts_rep/dpo.jsonl --out-dir $O/data/rcaw \
        $STUDY/data/caw_v1/caw.jsonl=CAW64 > $O/logs/rendercaw.log 2>&1
    mv $O/data/rcaw/CAW64.jsonl $O/data/CAW64.jsonl
    mv $O/data/rcaw/MANIFEST_render.json $O/data/MANIFEST_rendercaw.json
  fi
else
  for i in $(seq 1 120); do [ -f $O/data/MANIFEST_rendercaw.json ] && break; sleep 15; done
fi
python scripts/check_manifest.py $O/data/MANIFEST_rendercaw.json --file CAW64=$O/data/CAW64.jsonl --n CAW64=1433
for S in $SEEDS; do
  N=CAW64_s$S
  [ -f $O/runs/$N/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PIB --data $O/data/CAW64.jsonl \
      --lr 1e-4 --seed $S --out $O/runs/$N > $O/logs/train_$N.log 2>&1
  rm -rf $O/runs/$N/checkpoint-*
  (cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --prompts std --out $O/uo --runs $N=$O/runs/$N/adapter > $O/logs/uo_$N.log 2>&1)
  if [ ! -f $O/eval/$N.jsonl ]; then
    python scripts/evaluate_policies.py --model $PIB --adapter $O/runs/$N/adapter --name syn_$N --splits test,heldout --skip-probe > $O/logs/eval_$N.log 2>&1
    mv results/eval/syn_$N.jsonl $O/eval/$N.jsonl; mv results/eval/syn_$N.summary.json $O/eval/$N.summary.json
  fi
  (cd $PROJ && python scripts/eval_tau2_ac.py --model $PIB --out $T2 --runs $N=$O/runs/$N/adapter > $T2/logs/eval_$N.log 2>&1)
done
echo CAW_DONE
