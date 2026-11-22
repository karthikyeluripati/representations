#!/usr/bin/env bash
# Matched CF-SFT pilot (pre-registered: docs/cf_sft_pilot_protocol.md). NOT RUN: every stage needs explicit authorization.
#   STAGE=0              prompt selection on T2 DEV + baselines on T2 test / frozen T1 (no training)
#   STAGE=1 ARM=CF|CTRL  seed 0 training + diagnostic evals (futility gate)          [one pod per arm]
#   STAGE=2 ARM=.. SEED=..  remaining seeds; REG=1 adds BFCL/original/tau2 regression evals
# Hyperparameters are CAW-v2's exactly (LoRA r8, lr 1e-4, 3 epochs, batch 4, from pi_B); no search.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; R=$W/replication; PIB=$W/models/piB; RUNS=$W/synth/runs; O=$W/cf_pilot; T1F=$W/semcf_eval
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/cache $O/{runs,t2dev,t2test,t1,reg/eval,reg/uo,reg/tau2,logs,data}
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" "trl==0.24.0" pytest > $O/logs/pip.log 2>&1
pip install -q --no-deps "bfcl-eval==2026.3.23" >> $O/logs/pip.log 2>&1
pip install -q loguru docstring-parser deepdiff addict toml python-dotenv >> $O/logs/pip.log 2>&1
(cd $PIB && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: piB checksum mismatch"; exit 3; }
python -m pytest -q tests/test_cf_pilot.py tests/test_semantic_cf.py > $O/logs/tests.log 2>&1 || { tail -20 $O/logs/tests.log; exit 4; }
python - <<'EOF' || { echo "ABORT: frozen spec/data hash mismatch"; exit 5; }
import hashlib, json, sys
sys.path.insert(0, "src")
from verification import semantic_cf as T1, semantic_cf_t2 as T2
h = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
assert h(T1.specs("test")).startswith("a591d07a")
v = json.load(open("data/cf_sft_v1/validation_t2.json")); m = json.load(open("data/cf_sft_v1/MANIFEST.json"))
assert h(T2.specs("dev")) == v["dev"]["specs_sha256"] and h(T2.specs("test")) == v["test"]["specs_sha256"]
for a in ("CF", "CTRL", "ABL"):
    assert hashlib.sha256(open(f"data/cf_sft_v1/{a}.jsonl", "rb").read()).hexdigest() == m["arms"][a]["sha256"], a
EOF
EVAL="python scripts/eval_cf_pilot.py --model $PIB"
case "${STAGE:?}" in
0)
  $EVAL --template T2 --split dev --out $O/t2dev --runs piB@P0= piB@P1= piB@P2= piB@P3= \
      CAW2_s0@P0=$RUNS/CAW2_s0/adapter CAW2_s0@P1=$RUNS/CAW2_s0/adapter CAW2_s0@P2=$RUNS/CAW2_s0/adapter CAW2_s0@P3=$RUNS/CAW2_s0/adapter > $O/logs/stage0_dev.log 2>&1
  BP=$(python -c "import sys; sys.path.insert(0,'scripts'); sys.argv=['x']; import cf_pilot_analysis as A; print(A.best_prompt('$O/t2dev','piB')[0])")
  BC=$(python -c "import sys; sys.path.insert(0,'scripts'); sys.argv=['x']; import cf_pilot_analysis as A; print(A.best_prompt('$O/t2dev','CAW2_s0')[0])")
  echo "selected prompts: piB=$BP CAW2=$BC" | tee $O/logs/prompt_selection.txt
  $EVAL --template T2 --split test --out $O/t2test --runs piB@P0= piB@$BP= SYN64_s0@P0=$RUNS/SYN64IAR_s0/adapter SYN64_s1@P0=$RUNS/SYN64IAR_s1/adapter \
      SYN64_s2@P0=$RUNS/SYN64IAR_s2/adapter CAW_s0@P0=$RUNS/CAW64_s0/adapter CAW_s1@P0=$RUNS/CAW64_s1/adapter \
      CAW2_s0@P0=$RUNS/CAW2_s0/adapter CAW2_s1@P0=$RUNS/CAW2_s1/adapter CAW2_s0@$BC=$RUNS/CAW2_s0/adapter CAW2_s1@$BC=$RUNS/CAW2_s1/adapter > $O/logs/stage0_t2.log 2>&1
  $EVAL --template T1 --split test --out $O/t1 --runs piB@$BP= CAW2_s0@$BC=$RUNS/CAW2_s0/adapter CAW2_s1@$BC=$RUNS/CAW2_s1/adapter > $O/logs/stage0_t1.log 2>&1
  ;;
1|2)
  : "${ARM:?}"; SEED=${SEED:-0}; N=${ARM}_s$SEED
  if mkdir $O/data/render_$ARM.lock 2>/dev/null; then
    [ -f $O/data/MANIFEST_render_$ARM.json ] || { python scripts/render_sft_records.py --tokenizer $PIB --b-dpo $R/data/cts_rep/dpo.jsonl \
        --out-dir $O/data/r_$ARM $STUDY/data/cf_sft_v1/$ARM.jsonl=$ARM > $O/logs/render_$ARM.log 2>&1
      mv $O/data/r_$ARM/$ARM.jsonl $O/data/$ARM.jsonl; mv $O/data/r_$ARM/MANIFEST_render.json $O/data/MANIFEST_render_$ARM.json; }
  else
    for i in $(seq 1 120); do [ -f $O/data/MANIFEST_render_$ARM.json ] && break; sleep 15; done
  fi
  python scripts/check_manifest.py $O/data/MANIFEST_render_$ARM.json --file $ARM=$O/data/$ARM.jsonl --n $ARM=1433
  [ -f $O/runs/$N/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PIB --data $O/data/$ARM.jsonl \
      --lr 1e-4 --seed $SEED --out $O/runs/$N > $O/logs/train_$N.log 2>&1
  rm -rf $O/runs/$N/checkpoint-*
  $EVAL --template T2 --split test --out $O/t2test --runs $N@P0=$O/runs/$N/adapter > $O/logs/t2_$N.log 2>&1
  $EVAL --template T1 --split test --out $O/t1 --runs $N@P0=$O/runs/$N/adapter > $O/logs/t1_$N.log 2>&1
  if [ "${REG:-0}" = 1 ]; then
    # comparison traces for the S4 gate (existing CAW-v2 runs + the piB UO reference), copied, never regenerated
    cp -n $W/synth/eval/CAW2_s*.jsonl $O/reg/eval/; cp -n $W/synth/uo/CAW2_s*.jsonl $W/synth/uo/piB.jsonl $O/reg/uo/; cp -n $W/tau2_eval/CAW2_s*.jsonl $O/reg/tau2/
    (cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --prompts std --out $O/reg/uo --runs $N=$O/runs/$N/adapter > $O/logs/uo_$N.log 2>&1)
    if [ ! -f $O/reg/eval/$N.jsonl ]; then
      python scripts/evaluate_policies.py --model $PIB --adapter $O/runs/$N/adapter --name cfp_$N --splits test,heldout --skip-probe > $O/logs/eval_$N.log 2>&1
      mv results/eval/cfp_$N.jsonl $O/reg/eval/$N.jsonl; mv results/eval/cfp_$N.summary.json $O/reg/eval/$N.summary.json
    fi
    (cd $PROJ && python scripts/eval_tau2_ac.py --model $PIB --out $O/reg/tau2 --runs $N=$O/runs/$N/adapter > $O/logs/tau2_$N.log 2>&1)
  fi
  ;;
esac
echo CF_PILOT_STAGE_${STAGE}_DONE
