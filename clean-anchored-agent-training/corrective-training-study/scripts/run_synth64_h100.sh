#!/usr/bin/env bash
# Diversity-scaling study (pre-registered: docs/method_synthetic.md, "Diversity-scaling study"). SEEDS per pod.
# Pods share the volume: the first to take the lock renders the data; the others wait for its manifest.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; R=$W/replication; PIB=$W/models/piB; O=$W/synth
SEEDS=${SEEDS:-"0"}; TAG=64_$(echo $SEEDS | tr " " "_")
CLT="MessageAPI,TradingBot,TwitterAPI,TicketAPI"
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CTS_TOOL_ANNOTATIONS
mkdir -p /root/cache $O/{runs,uo,eval,logs,data} logs $PROJ/logs
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" "trl==0.24.0" pytest hf_transfer
pip install -q --no-deps "bfcl-eval==2026.3.23"
(cd $PIB && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: piB checksum mismatch"; exit 3; }
(cd $PROJ && python -m pytest -q tests) > $O/logs/tests_parent_$TAG.log 2>&1 || { tail -20 $O/logs/tests_parent_$TAG.log; exit 4; }
python -m pytest -q tests/test_synth_data.py > $O/logs/tests_synth_$TAG.log 2>&1 || { tail -20 $O/logs/tests_synth_$TAG.log; exit 4; }
python - <<PY
import json
s=json.load(open("$STUDY/data/synth_v2_64/MANIFEST.json")); assert s["n"]==832 and s["n_classes"]==64 and not s["dropped"], s
m=json.load(open("$STUDY/data/iar_v1/MANIFEST.json")); assert m["n"]==517 and not m["dropped"], m
print("data ok")
PY
if mkdir $O/data/render64.lock 2>/dev/null; then
  [ -f $O/data/MANIFEST_render64.json ] || { cat $STUDY/data/iar_v1/iar.jsonl $STUDY/data/synth_v2_64/syn_iar.jsonl > $O/data/syn64IAR_abstract.jsonl
    python scripts/render_sft_records.py --tokenizer $PIB --b-dpo $R/data/cts_rep/dpo.jsonl --out-dir $O/data/r64 \
        $O/data/syn64IAR_abstract.jsonl=SYN64IAR > $O/logs/render64.log 2>&1
    mv $O/data/r64/SYN64IAR.jsonl $O/data/SYN64IAR.jsonl; mv $O/data/r64/MANIFEST_render.json $O/data/MANIFEST_render64.json; }
else
  for i in $(seq 1 120); do [ -f $O/data/MANIFEST_render64.json ] && break; sleep 15; done
fi
[ -f $O/data/MANIFEST_render64.json ] || { echo "ABORT: render64 not available"; exit 5; }
python - <<PY
import hashlib, json
m = json.load(open("$O/data/MANIFEST_render64.json"))
assert hashlib.sha256(open("$O/data/SYN64IAR.jsonl", "rb").read()).hexdigest() == m["SYN64IAR"]["sha256"]; print("render64 ok", m["SYN64IAR"]["n"])
PY
for S in $SEEDS; do
  N=SYN64IAR_s$S
  [ -f $O/runs/$N/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PIB --data $O/data/SYN64IAR.jsonl \
      --lr 1e-4 --seed $S --out $O/runs/$N > $O/logs/train_$N.log 2>&1
  rm -rf $O/runs/$N/checkpoint-*
  (cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --prompts std --out $O/uo --runs $N=$O/runs/$N/adapter > $O/logs/uo_$N.log 2>&1)
  [ -f $O/eval/$N.jsonl ] || { python scripts/evaluate_policies.py --model $PIB --adapter $O/runs/$N/adapter --name syn_$N \
      --splits test,heldout --skip-probe > $O/logs/eval_$N.log 2>&1
    mv results/eval/syn_$N.jsonl $O/eval/$N.jsonl; mv results/eval/syn_$N.summary.json $O/eval/$N.summary.json; }
done
echo SYN64_DONE
