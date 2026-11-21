#!/usr/bin/env bash
# Two matched baselines (A balanced SFT, B CSO-style verified DPO) from a rebuilt pi0 on ONE H100 80GB.
# Requires explicit GPU authorization. Resumable: every step skips completed outputs. Phases: PHASE=setup|pi0|bench|dev|final|all
set -euo pipefail
cd "$(dirname "$0")/.."                       # corrective-training-study/
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=${WORK:-/workspace}; BASE=$W/models/Qwen3-4B-Instruct-2507; PI0=$W/models/pi0; REV=cdbee75f17c01a7cc42f958dc650907174af0554
ARCH=${ARCHIVE:-$W/pod_backup_20261119.tgz}; PHASE=${PHASE:-all}
export PYTHONHASHSEED=0 PIP_CACHE_DIR=$W/cache/pip HF_HOME=$W/cache/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $W/{cache/pip,cache/hf,tmp,models} logs results/runs $PROJ/logs
date -u +%s > logs/session_start_$(date -u +%Y%m%dT%H%M%S)
run() { [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }

if run setup; then
  [ -d $W/venv ] || python3 -m venv --system-site-packages $W/venv
  source $W/venv/bin/activate
  pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" "trl==0.24.0" pytest certifi hf_transfer
  pip install -q --no-deps "bfcl-eval==2026.3.23"
  [ -d $BASE ] || HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV', local_dir='$BASE')"
  (cd $PROJ && python -m pytest -q tests) | tee logs/tests_parent.log
  CTS_TOKENIZER=$BASE python -m pytest -q tests | tee logs/tests_study.log
  (cd $PROJ && python scripts/validate_environments.py > logs/validate.log && python scripts/construct_correction_pairs.py > logs/construct.log)
  python - <<EOF
import json; m=json.load(open("$PROJ/data/cacrt_pairs_v1/MANIFEST.json")); r=json.load(open("$PROJ/data/cacrt_pairs_v1/MANIFEST.local.json"))
assert m["files"]==r["files"] and m["tasks_file_sha256"]==r["tasks_file_sha256"], "benchmark data differ"; print("benchmark hashes match")
EOF
  python scripts/check_archive.py --archive $ARCH | tee logs/check_archive.log
  [ -f data/cts_v1/MANIFEST.json ] || python scripts/build_cso_pairs.py --tokenizer $BASE | tee logs/build.log
fi
source $W/venv/bin/activate

if run pi0; then   # rebuild the starting policy with the archived STAGE0 config, then the reproduction check
  [ -d $PI0 ] || (cd $PROJ && python scripts/train_balanced_sft.py --arm STAGE0 --model $BASE --lora --lr 1e-4 --epochs 1 \
       --batch-size 8 --micro-batch 1 --max-len 12288 --save --merge-out $PI0 2>&1 | tee $STUDY/logs/stage0.log)
  python scripts/evaluate_policies.py --model $PI0 --name pi0_rebuilt --splits test,heldout --skip-probe | tee logs/eval_pi0_rebuilt.log
  set +e; python scripts/compare_reproduction.py --rebuilt results/eval/pi0_rebuilt.jsonl --archive $ARCH --out results/pi0_reproduction.json; rc=$?; set -e
  if [ $rc -eq 3 ]; then
    echo "pi0 not reproduced within tolerance -> regenerating verified pairs under the rebuilt pi0" | tee -a logs/decisions.log
    (cd $PROJ && { mv data/cacrt_pairs_v1_policy data/cacrt_pairs_v1_policy.archived_pi0 2>/dev/null || true; }; python scripts/evaluate.py --mode sample_states --model $PI0 --k 4 | tee $STUDY/logs/sample_states.log)
    cp $PROJ/data/cacrt_pairs_v1_policy/train_policy_pairs.jsonl data/policy_pairs_v1_rebuilt.jsonl
    python scripts/build_cso_pairs.py --tokenizer $BASE --pairs data/policy_pairs_v1_rebuilt.jsonl --out data/cts_v1_rebuilt | tee logs/build_rebuilt.log
    echo "DATA=cts_v1_rebuilt" > logs/data_choice.env
  else echo "DATA=cts_v1" > logs/data_choice.env; fi
  python scripts/evaluate_policies.py --model $PI0 --name pi0_dev --splits dev | tee logs/eval_pi0_dev.log
  python scripts/evaluate_policies.py --model $PI0 --name pi0_probe_testheld --splits test,heldout | tee logs/probe_pi0.log
fi
source logs/data_choice.env 2>/dev/null || DATA=cts_v1

if run bench; then   # measured throughput -> projected budget (abort rule in configs/baselines.json)
  python scripts/train_verified_dpo.py --model $PI0 --data data/$DATA/dpo.jsonl --max-steps-bench 3 --out results/runs/bench_dpo
  python scripts/train_balanced_sft.py --model $PI0 --data data/$DATA/sft_balanced.jsonl --max-steps-bench 3 --out results/runs/bench_sft
  python - <<'EOF' | tee logs/budget_projection.log
import json
d=json.load(open("results/runs/bench_dpo/timing.json"))["train_seconds"]/3; s=json.load(open("results/runs/bench_sft/timing.json"))["train_seconds"]/3
steps=567*3//4
h=(2*d*steps+2*s*steps)/3600 + 7*0.1 + 0.6
print(json.dumps({"dpo_s_per_step":d,"sft_s_per_step":s,"steps_per_run":steps,"projected_total_h_incl_evals":round(h,2),"within_5h":h<=5}))
EOF
fi

if run dev; then   # lr grid, selection on dev only
  for LR in 5e-6 5e-5; do [ -f results/runs/dpo_lr$LR/adapter/adapter_config.json ] || python scripts/train_verified_dpo.py --model $PI0 --data data/$DATA/dpo.jsonl --lr $LR --out results/runs/dpo_lr$LR 2>&1 | tee logs/train_dpo_lr$LR.log
    python scripts/evaluate_policies.py --model $PI0 --adapter results/runs/dpo_lr$LR/adapter --name dpo_lr${LR}_dev --splits dev --skip-probe; done
  for LR in 1e-5 1e-4; do [ -f results/runs/sft_lr$LR/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PI0 --data data/$DATA/sft_balanced.jsonl --lr $LR --out results/runs/sft_lr$LR 2>&1 | tee logs/train_sft_lr$LR.log
    python scripts/evaluate_policies.py --model $PI0 --adapter results/runs/sft_lr$LR/adapter --name sft_lr${LR}_dev --splits dev --skip-probe; done
  python - <<'EOF' | tee logs/dev_selection.log
import json
def pick(names):
    sc={n: json.load(open(f"results/eval/{n}_dev.summary.json"))["dev"] for n in names}
    best=max(sc, key=lambda n:(round(sc[n]["overall"],3), -sc[n]["false_outage_rate"]))
    return best, {n:{k:sc[n][k] for k in ("overall","false_outage_rate","fault_free","genuine_repair")} for n in sc}
b=pick(["dpo_lr5e-6","dpo_lr5e-5"]); a=pick(["sft_lr1e-5","sft_lr1e-4"])
json.dump({"B_dpo":b,"A_sft":a}, open("results/dev_selection.json","w"), indent=1); print(json.dumps({"B_dpo":b,"A_sft":a}, indent=1))
EOF
fi

if run final; then  # once per selected frozen adapter
  for SEL in $(python -c "import json;d=json.load(open('results/dev_selection.json'));print(d['A_sft'][0],d['B_dpo'][0])"); do
    python scripts/evaluate_policies.py --model $PI0 --adapter results/runs/$SEL/adapter --name ${SEL}_final --splits test,heldout | tee logs/eval_${SEL}_final.log; done
  (cd $PROJ && python scripts/evaluate.py --mode summarize --ref $STUDY/results/eval/pi0_rebuilt.jsonl --files $STUDY/results/eval/*_final.jsonl) > results/final_vs_pi0.json
  A=$(python -c "import json;print(json.load(open('results/dev_selection.json'))['A_sft'][0])")
  (cd $PROJ && python scripts/evaluate.py --mode summarize --ref $STUDY/results/eval/${A}_final.jsonl --files $STUDY/results/eval/$(python -c "import json;print(json.load(open('$STUDY/results/dev_selection.json'))['B_dpo'][0])")_final.jsonl) > results/final_B_vs_A.json
fi
# Backup (run, then copy OFF the pod and verify sha256): everything except the 8 GB models.
tar --exclude=__pycache__ -czf $W/cts_backup_$(date -u +%Y%m%d).tgz -C $PROJ/.. clean-anchored-agent-training/corrective-training-study clean-anchored-agent-training/results clean-anchored-agent-training/data
sha256sum $W/cts_backup_*.tgz | tee logs/backup_sha256.txt
