#!/usr/bin/env bash
# IAR with MCP-style idempotency annotations (pre-registered: docs/method_iar_annotations.md). Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; R=$W/replication; PI0=$W/models/pi0; PIB=$W/models/piB; O=$W/iar_ann; S=0
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
(cd $PROJ && python scripts/feasibility_ambiguous_commit.py > $O/logs/feasibility.log 2>&1 && python scripts/label_idempotency.py > $O/logs/labels.log 2>&1)
python scripts/build_iar_data.py > $O/logs/build.log 2>&1
python - <<EOF
import json; m=json.load(open("$STUDY/data/iar_v1/MANIFEST.json")); assert m["n"]==517 and not m["dropped"], m
a=json.load(open("$PROJ/results/tool_annotations.json")); assert len(a)==78 and not a["book_flight"]["idempotentHint"]; print("data+labels ok")
EOF
export CTS_TOOL_ANNOTATIONS=$PROJ/results/tool_annotations.json        # every render and evaluation below sees annotations
[ -f $O/data/MANIFEST_render.json ] || python scripts/render_sft_records.py --tokenizer $PIB --b-dpo $R/data/cts_rep/dpo.jsonl \
    --out-dir $O/data $STUDY/data/iar_v1/iar.jsonl=IARA $STUDY/data/iar_v1/aac_matched.jsonl=AAC3A > $O/logs/render.log 2>&1
grep -q "idempotentHint" $O/data/IARA.jsonl || { echo "ABORT: annotations missing from rendered prompts"; exit 6; }
[ -f $O/uo/piB.jsonl ] || cp $W/otv/uo/piB.jsonl $W/otv/uo/piB.summary.json $O/uo/
for ARM in AAC3A IARA; do
  N=${ARM}_s$S
  [ -f $O/runs/$N/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PIB --data $O/data/$ARM.jsonl \
      --lr 1e-4 --seed $S --out $O/runs/$N > $O/logs/train_$N.log 2>&1
  rm -rf $O/runs/$N/checkpoint-*
done
(cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --prompts std --out $O/uo \
    --runs AAC3A_s0=$O/runs/AAC3A_s0/adapter IARA_s0=$O/runs/IARA_s0/adapter IARTx_s0=$W/iar/runs/IART_s0/adapter piBA_s0= > $O/logs/uo.log 2>&1)
for N in AAC3A_s0 IARA_s0; do
  [ -f $O/eval/$N.jsonl ] || { python scripts/evaluate_policies.py --model $PIB --adapter $O/runs/$N/adapter --name iara_$N \
      --splits test,heldout --skip-probe > $O/logs/eval_$N.log 2>&1
    mv results/eval/iara_$N.jsonl $O/eval/$N.jsonl; mv results/eval/iara_$N.summary.json $O/eval/$N.summary.json; }
done
python scripts/otv_analysis.py --uo $O/uo --arms IARA AAC3A IARTx piBA --method IARA --train-classes $CLT --seeds 0 \
    --out $O/analysis.json > $O/logs/analysis.log
tar --exclude=__pycache__ -czf $W/iar_ann_slim.tgz -C $W iar_ann/uo iar_ann/eval iar_ann/logs iar_ann/analysis.json \
    $(cd $W && ls iar_ann/data/MANIFEST*.json iar_ann/runs/*/run_config.json iar_ann/runs/*/log_history.json 2>/dev/null)
sha256sum $W/iar_ann_slim.tgz
echo IARA_DONE
