#!/usr/bin/env bash
# Replication of Baselines A (balanced SFT) and B (verified DPO) over seeds 0,1,2 from ONE persisted pi0.
# Pre-registered (2026-11-19, before any replication result):
#   * lr fixed from the seed-0 run (abort rule): B 5e-6 (CSO default), A 1e-4; all other settings as configs/baselines.json.
#   * pi0 is rebuilt ONCE and stored on the network volume; pairs are re-verified under it (K=4) before any training.
#   * primary: held-out overall success, seed-pooled paired cluster bootstrap (scripts/pool_seeds.py), B - A and each - pi0.
#   * no model selection; every final adapter is evaluated once on test+heldout.
# Requires: pod started WITH a RunPod network volume mounted at /workspace. Refuses to run otherwise.
# Resumable: every step skips completed outputs. SEEDS can be overridden (e.g. SEEDS="0").
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; BASE=/root/models/Qwen3-4B-Instruct-2507;   # public base: container disk (re-downloadable); pi0+results: volume
 PI0=$W/models/pi0; REV=cdbee75f17c01a7cc42f958dc650907174af0554
SEEDS=${SEEDS:-"0 1 2"}; R=$W/replication                 # all outputs live on the network volume
df -h $W | tail -1 | grep -q -E "mfs|runpod.net|nfs" || { echo "ABORT: /workspace is not a network volume; pi0 would not persist"; exit 2; }
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf TMPDIR=/tmp HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/models /root/cache $W/models $R/{runs,eval,logs} logs $PROJ/logs
date -u +%s >> $R/logs/session_starts

# --- environment (venv on the container disk: fast, rebuilt per pod; models and results on the volume) ---
VENV=/root/venv
[ -d $VENV ] || python3 -m venv --system-site-packages $VENV
source $VENV/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" "trl==0.24.0" pytest certifi hf_transfer
pip install -q --no-deps "bfcl-eval==2026.3.23"
[ -f $BASE/config.json ] || HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV', local_dir='$BASE')"
(cd $PROJ && python -m pytest -q tests) > $R/logs/tests_parent.log
CTS_TOKENIZER=$BASE python -m pytest -q tests > $R/logs/tests_study.log
(cd $PROJ && python scripts/validate_environments.py > logs/validate.log && python scripts/construct_correction_pairs.py > logs/construct.log)
python - <<EOF
import json; m=json.load(open("$PROJ/data/cacrt_pairs_v1/MANIFEST.json")); r=json.load(open("$PROJ/data/cacrt_pairs_v1/MANIFEST.local.json"))
assert m["files"]==r["files"] and m["tasks_file_sha256"]==r["tasks_file_sha256"], "benchmark data differ"; print("benchmark hashes match")
EOF

# --- pi0: rebuilt once, persisted, hashed ---
if [ ! -f $PI0/DONE ]; then
  (cd $PROJ && python scripts/train_balanced_sft.py --arm STAGE0 --model $BASE --lora --lr 1e-4 --epochs 1 --batch-size 8 \
      --micro-batch 1 --max-len 12288 --save --merge-out $PI0 --out-root $R/runs > $R/logs/stage0.log 2>&1)
  (cd $PI0 && sha256sum *.safetensors > SHA256SUMS) && touch $PI0/DONE
fi
(cd $PI0 && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: pi0 checksum mismatch"; exit 3; }
[ -f $R/eval/pi0.jsonl ] || { python scripts/evaluate_policies.py --model $PI0 --name pi0 --splits test,heldout --skip-probe > $R/logs/eval_pi0.log 2>&1;
                              mv results/eval/pi0.jsonl results/eval/pi0.summary.json $R/eval/; }

# --- pairs re-verified under THIS pi0; datasets built ---
if [ ! -f $R/data/cts_rep/MANIFEST.json ]; then
  (cd $PROJ && rm -rf data/cacrt_pairs_v1_policy && python scripts/evaluate.py --mode sample_states --model $PI0 --k 4 > $R/logs/sample_states.log 2>&1)
  mkdir -p $R/data && cp -r $PROJ/data/cacrt_pairs_v1_policy $R/data/ && cp $PROJ/data/cacrt_pairs_v1_policy/train_policy_pairs.jsonl $R/data/policy_pairs.jsonl
  python scripts/build_cso_pairs.py --tokenizer $BASE --pairs $R/data/policy_pairs.jsonl --out $R/data/cts_rep > $R/logs/build.log
fi

# --- arms x seeds ---
for S in $SEEDS; do
  [ -f $R/runs/B_s$S/adapter/adapter_config.json ] || python scripts/train_verified_dpo.py --model $PI0 --data $R/data/cts_rep/dpo.jsonl \
      --lr 5e-6 --seed $S --precompute-ref --out $R/runs/B_s$S > $R/logs/train_B_s$S.log 2>&1
  [ -f $R/runs/A_s$S/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PI0 --data $R/data/cts_rep/sft_balanced.jsonl \
      --lr 1e-4 --seed $S --out $R/runs/A_s$S > $R/logs/train_A_s$S.log 2>&1
  for ARM in A B; do
    rm -rf $R/runs/${ARM}_s$S/checkpoint-*                   # final adapter kept; optimizer checkpoints dropped once complete
    [ -f $R/eval/${ARM}_s$S.jsonl ] || { python scripts/evaluate_policies.py --model $PI0 --adapter $R/runs/${ARM}_s$S/adapter \
        --name ${ARM}_s$S --splits test,heldout > $R/logs/eval_${ARM}_s$S.log 2>&1
      mv results/eval/${ARM}_s$S.jsonl results/eval/${ARM}_s$S.summary.json $R/eval/; cp results/probe/${ARM}_s$S.json* $R/eval/ 2>/dev/null || true
      cp results/failure_types/${ARM}_s$S.json $R/eval/ 2>/dev/null || true; }
  done
done

# --- pooled analysis + backup archive on the volume ---
FA=$(for S in $SEEDS; do echo -n "$R/eval/A_s$S.jsonl "; done); FB=$(for S in $SEEDS; do echo -n "$R/eval/B_s$S.jsonl "; done)
python scripts/pool_seeds.py --arm A $FA --arm B $FB --ref $R/eval/pi0.jsonl --out $R/pooled.json | tee $R/logs/pooled.log
tar --exclude=__pycache__ -czf $R/replication_results.tgz -C $W replication/eval replication/data replication/logs replication/pooled.json \
    $(cd $W && ls -d replication/runs/*/adapter replication/runs/*/run_config.json replication/runs/*/log_history.json 2>/dev/null)
sha256sum $R/replication_results.tgz | tee $R/logs/backup_sha256.txt
echo REPLICATION_DONE
