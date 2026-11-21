#!/usr/bin/env bash
# Synthetic-API diversity study (pre-registered: docs/method_synthetic.md). Resumable; SEEDS default "0".
# Several pods may share the volume with different SEEDS: data are built only if absent (deterministic) and every
# per-seed output and log carries the seed in its name.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; R=$W/replication; PI0=$W/models/pi0; PIB=$W/models/piB; O=$W/synth
SEEDS=${SEEDS:-"0"}; TAG=$(echo $SEEDS | tr " " "_")
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
python - <<EOF
import json
m=json.load(open("$STUDY/data/iar_v1/MANIFEST.json")); assert m["n"]==517 and not m["dropped"], m
s=json.load(open("$STUDY/data/synth_v1/MANIFEST.json")); assert s["n"]==832 and not s["dropped"], s
print("data ok")
EOF
for ARM in SYNIAR SYNOLD; do                              # rendered once by the seed-0 run; must be unchanged
  [ -f $O/data/$ARM.jsonl ] || { echo "ABORT: $O/data/$ARM.jsonl missing (render with the seed-0 run first)"; exit 5; }
done
python - <<EOF
import hashlib, json
m = json.load(open("$O/data/MANIFEST_render.json"))
for k in ("SYNIAR", "SYNOLD"):
    assert hashlib.sha256(open(f"$O/data/{k}.jsonl", "rb").read()).hexdigest() == m[k]["sha256"], k
print("rendered data unchanged")
EOF
for S in $SEEDS; do
  for ARM in SYNOLD SYNIAR; do
    N=${ARM}_s$S
    [ -f $O/runs/$N/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PIB --data $O/data/$ARM.jsonl \
        --lr 1e-4 --seed $S --out $O/runs/$N > $O/logs/train_$N.log 2>&1
    rm -rf $O/runs/$N/checkpoint-*
  done
  (cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --prompts std --out $O/uo \
      --runs SYNOLD_s$S=$O/runs/SYNOLD_s$S/adapter SYNIAR_s$S=$O/runs/SYNIAR_s$S/adapter > $O/logs/uo_s$S.log 2>&1)
  for N in SYNOLD_s$S SYNIAR_s$S; do
    [ -f $O/eval/$N.jsonl ] || { python scripts/evaluate_policies.py --model $PIB --adapter $O/runs/$N/adapter --name syn_$N \
        --splits test,heldout --skip-probe > $O/logs/eval_$N.log 2>&1
      mv results/eval/syn_$N.jsonl $O/eval/$N.jsonl; mv results/eval/syn_$N.summary.json $O/eval/$N.summary.json; }
  done
done
DS=$(for f in $O/uo/SYNIAR_s*.jsonl; do basename $f .jsonl | sed 's/SYNIAR_s//'; done | sort | tr '\n' ' ')
python scripts/otv_analysis.py --uo $O/uo --arms SYNIAR SYNOLD --method SYNIAR --train-classes $CLT --seeds $DS \
    --out $O/analysis_$TAG.json > $O/logs/analysis_$TAG.log
tar --exclude=__pycache__ -czf $W/synth_slim_$TAG.tgz -C $W synth/uo synth/eval synth/logs $(cd $W && ls synth/analysis*.json) \
    $(cd $W && ls synth/data/MANIFEST*.json synth/runs/*/run_config.json synth/runs/*/log_history.json 2>/dev/null)
sha256sum $W/synth_slim_$TAG.tgz
echo SYNTH_DONE
