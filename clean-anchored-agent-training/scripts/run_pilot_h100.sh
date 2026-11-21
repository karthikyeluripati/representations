#!/usr/bin/env bash
# Three-arm CACRT pilot on ONE H100 80GB. Do not run without explicit GPU authorization.
# Every step is resumable: it skips work whose output already exists. Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
# Controls (authorization-bounded): SEEDS (default "0" = seed-0 gate only), PHASE=gate|arms|all, STEPS (180, or the
# pre-registered reduced 120 if measured throughput is slower than 1.5 s per sequence).
SEEDS=${SEEDS:-0}; PHASE=${PHASE:-all}; STEPS=${STEPS:-180}
[ -f /workspace/env.sh ] && source /workspace/env.sh
export PYTHONHASHSEED=0   # BFCL error strings print Python sets; fixed hash seed makes observations byte-reproducible
W=${WORK:-/workspace}                 # large volume (>= 60 GB): models, adapters, eval traces
BASE=$W/models/Qwen3-4B-Instruct-2507
PI0=$W/models/pi0
REV=cdbee75f17c01a7cc42f958dc650907174af0554
mkdir -p logs "$W/models" results/runs results/eval
T0=$(date +%s); echo "start $(date -u)" > logs/gpu_session_start

if [ "$PHASE" = "arms" ]; then SKIP_SETUP=1; fi
if [ -z "${SKIP_SETUP:-}" ]; then
# 0. environment (pinned)
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" pytest
pip install -q --no-deps "bfcl-eval==2026.3.23"
[ -d "$BASE" ] || huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --revision $REV --local-dir "$BASE"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee logs/gpu.txt

# 1. CPU checks on the pod (must reproduce the local hashes)
python -m pytest -q tests | tee logs/tests.log
python scripts/validate_environments.py > logs/validate.log
python scripts/construct_correction_pairs.py > logs/construct.log
python - <<'EOF'
import json; m=json.load(open("data/cacrt_pairs_v1/MANIFEST.json")); ref=json.load(open("data/cacrt_pairs_v1/MANIFEST.local.json"))
assert m["files"]==ref["files"], "pair data differ from the locally validated build"
print("pair data hashes match")
EOF

# 2. stage-0 policy pi0 (base + success + outage-recovery SFT), merged
[ -d "$PI0" ] || python scripts/train_balanced_sft.py --arm STAGE0 --model "$BASE" --lora --lr 1e-4 --epochs 1 \
     --batch-size 8 --micro-batch 1 --max-len 12288 --save --merge-out "$PI0" 2>&1 | tee logs/stage0.log

# 3. reference evaluations: frozen base (B0) and pi0
for M in base:$BASE pi0:$PI0; do n=${M%%:*}; p=${M#*:}
  [ -f results/eval/$n.jsonl ] || python scripts/evaluate.py --mode run --model "$p" --name $n 2>&1 | tee logs/eval_$n.log; done

# 4. premise gate (pre-registered): do corrective DPO gradients conflict more with twins than with generic anchors?
python scripts/analyze_interference.py --mode premise --model "$PI0" --lora --n 64 --batch 8 --rounds 16 2>&1 | tee logs/premise.log

# 5. policy-sampled negatives + policy-continuation verification on train states (CSO-style)
[ -f data/cacrt_pairs_v1_policy/MANIFEST.json ] || python scripts/evaluate.py --mode sample_states --model "$PI0" --k 4 2>&1 | tee logs/sample_states.log

if [ "$PHASE" = "gate" ]; then echo "gate phase done; inspect logs/premise.log, data/cacrt_pairs_v1_policy/MANIFEST.json"; exit 0; fi

fi  # end of setup/gate block
# 6. pilot arms x SEEDS (matched: same init, steps, batch, lr; anchors where the arm has them)
# pre-registered pair-source rule (configs/pilot.json)
PAIRS=$(python -c "import json;m=json.load(open('data/cacrt_pairs_v1_policy/MANIFEST.json'));import collections;import sys
rows=[json.loads(l) for l in open('data/cacrt_pairs_v1_policy/train_policy_pairs.jsonl')]
print('policy' if len({r['state'] for r in rows})>=100 else 'surrogate')")
echo "pair source: $PAIRS" | tee logs/pair_source.txt
COMMON="--pairs $PAIRS --policy-correct-anchors --model $PI0 --lora --lr 2e-5 --beta 0.1 --steps $STEPS --batch-size 8 --anchor-batch-size 8 --micro-batch 4 --max-len 12288 --val --eval-every 60 --save"
for S in $SEEDS; do
  for A in EXP1:train_balanced_sft EXP2:train_localized EXP3:train_clean_anchored B7_replay:train_clean_anchored; do
    arm=${A%%:*}; sc=${A#*:}
    [ -d results/runs/${arm}_s$S/adapter ] || python scripts/$sc.py --arm $arm --seed $S $COMMON 2>&1 | tee logs/train_${arm}_s$S.log
    [ -f results/eval/${arm}_s$S.jsonl ] || python scripts/evaluate.py --mode run --model "$PI0" --adapter results/runs/${arm}_s$S/adapter --name ${arm}_s$S 2>&1 | tee logs/eval_${arm}_s$S.log
  done
done

# 7. analysis (all seeds present)
python scripts/analyze_interference.py --mode curves > logs/curves.log
python scripts/evaluate.py --mode summarize --ref results/eval/pi0.jsonl --files results/eval/*_s?.jsonl results/eval/base.jsonl > results/eval/pilot_summary_vs_pi0.json
echo "elapsed_s $(( $(date +%s) - T0 ))" | tee logs/gpu_session_elapsed
# Back up OFF the pod before terminating: results/, logs/, data/, adapters (results/runs/*/adapter, ~130 MB each at r=32).
