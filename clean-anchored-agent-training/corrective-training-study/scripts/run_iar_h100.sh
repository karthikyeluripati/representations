#!/usr/bin/env bash
# Idempotency-aware recovery study (pre-registered: docs/method_iar.md). Resumable; SEEDS default "0".
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; R=$W/replication; PI0=$W/models/pi0; PIB=$W/models/piB; O=$W/iar; SEEDS=${SEEDS:-"0"}
CLT="MessageAPI,TradingBot,TwitterAPI,TicketAPI"
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/cache $O/{runs,uo,eval,logs,data} logs $PROJ/logs
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" "trl==0.24.0" pytest hf_transfer
pip install -q --no-deps "bfcl-eval==2026.3.23"
(cd $PIB && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: piB checksum mismatch"; exit 3; }
(cd $PROJ && python -m pytest -q tests) > $O/logs/tests_parent.log 2>&1 || { tail -20 $O/logs/tests_parent.log; exit 4; }
(cd $PROJ && python scripts/feasibility_ambiguous_commit.py > $O/logs/feasibility.log 2>&1)
python scripts/build_iar_data.py > $O/logs/build.log 2>&1
python - <<EOF
import json; m=json.load(open("$STUDY/data/iar_v1/MANIFEST.json")); assert m["n"]==517 and not m["dropped"], m; print("iar data ok")
EOF
CTS_TOKENIZER=$PI0 python -m pytest -q tests > $O/logs/tests_study.log 2>&1 || { tail -20 $O/logs/tests_study.log; exit 4; }
[ -f $O/data/MANIFEST_render.json ] || python scripts/render_sft_records.py --tokenizer $PIB --b-dpo $R/data/cts_rep/dpo.jsonl \
    --out-dir $O/data $STUDY/data/iar_v1/iar.jsonl=IART $STUDY/data/iar_v1/aac_matched.jsonl=AAC3T > $O/logs/render.log 2>&1
[ -f $O/uo/piB.jsonl ] || cp $W/otv/uo/piB.jsonl $W/otv/uo/piB.summary.json $O/uo/
for S in $SEEDS; do
  RUNS=""
  for ARM in AAC3T IART; do
    N=${ARM}_s$S
    [ -f $O/runs/$N/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PIB --data $O/data/$ARM.jsonl \
        --lr 1e-4 --seed $S --out $O/runs/$N > $O/logs/train_$N.log 2>&1
    rm -rf $O/runs/$N/checkpoint-*
    RUNS="$RUNS $N=$O/runs/$N/adapter"
  done
  (cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --prompts std --out $O/uo --runs $RUNS > $O/logs/uo_s$S.log 2>&1)
  for ARM in AAC3T IART; do
    N=${ARM}_s$S
    [ -f $O/eval/$N.jsonl ] || { python scripts/evaluate_policies.py --model $PIB --adapter $O/runs/$N/adapter --name iar_$N \
        --splits test,heldout --skip-probe > $O/logs/eval_$N.log 2>&1
      mv results/eval/iar_$N.jsonl $O/eval/$N.jsonl; mv results/eval/iar_$N.summary.json $O/eval/$N.summary.json; }
  done
done
DS=$(for f in $O/uo/IART_s*.jsonl; do basename $f .jsonl | sed 's/IART_s//'; done | sort | tr '\n' ' ')
python scripts/otv_analysis.py --uo $O/uo --arms IART AAC3T --method IART --train-classes $CLT --seeds $DS --out $O/analysis.json > $O/logs/analysis.log
tar --exclude=__pycache__ -czf $W/iar_slim.tgz -C $W iar/uo iar/eval iar/logs iar/analysis.json \
    $(cd $W && ls iar/data/MANIFEST*.json iar/runs/*/run_config.json iar/runs/*/log_history.json 2>/dev/null)
sha256sum $W/iar_slim.tgz
echo IAR_DONE
