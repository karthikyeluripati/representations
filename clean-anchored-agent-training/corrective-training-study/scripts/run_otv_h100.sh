#!/usr/bin/env bash
# OTV method study (pre-registered: docs/method_outcome_twins.md incl. edit logs). Resumable; SEEDS default "0".
# After seed 0 the futility gate is evaluated; seeds 1-2 are launched separately (SEEDS="1 2") only if it passes.
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; R=$W/replication; PI0=$W/models/pi0; PIB=$W/models/piB; O=$W/otv; SEEDS=${SEEDS:-"0"}
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/cache $O/{runs,uo,eval,logs,data} logs $PROJ/logs
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" "trl==0.24.0" pytest hf_transfer
pip install -q --no-deps "bfcl-eval==2026.3.23"
(cd $PI0 && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: pi0 checksum mismatch"; exit 3; }
(cd $PROJ && python -m pytest -q tests) > $O/logs/tests_parent.log 2>&1 || { tail -20 $O/logs/tests_parent.log; exit 4; }
(cd $PROJ && python scripts/feasibility_ambiguous_commit.py > $O/logs/feasibility.log 2>&1)
python scripts/build_otv_pairs.py > $O/logs/build_pairs.log 2>&1
python - <<EOF
import json; m=json.load(open("$STUDY/data/otv_v1/MANIFEST.json")); assert m["n_pairs"]==448, m["n_pairs"]; print("pairs ok")
EOF
CTS_TOKENIZER=$PI0 python -m pytest -q tests > $O/logs/tests_study.log 2>&1 || { tail -20 $O/logs/tests_study.log; exit 4; }

# pi_B = pi0 + B_s0, merged once, persisted, hashed
if [ ! -f $PIB/DONE ]; then
  python scripts/merge_adapter.py --base $PI0 --adapter $R/runs/B_s0/adapter --out $PIB > $O/logs/merge.log 2>&1
  (cd $PIB && sha256sum *.safetensors > SHA256SUMS) && touch $PIB/DONE
fi
(cd $PIB && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: piB checksum mismatch"; exit 3; }
[ -f $O/data/MANIFEST.json ] || python scripts/render_otv_arms.py --tokenizer $PIB --b-dpo $R/data/cts_rep/dpo.jsonl --out $O/data > $O/logs/render.log 2>&1
# pi_B reference evaluation on the uncertain-outcome suites (std prompt), once
(cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --runs piB= --prompts std --out $O/uo > $O/logs/uo_piB.log 2>&1)

for S in $SEEDS; do
  [ -f $O/runs/OTV_s$S/adapter/adapter_config.json ] || python scripts/train_verified_dpo.py --model $PIB --data $O/data/OTV.jsonl \
      --lr 5e-6 --seed $S --precompute-ref --out $O/runs/OTV_s$S > $O/logs/train_OTV_s$S.log 2>&1
  [ -f $O/runs/BAC_s$S/adapter/adapter_config.json ] || python scripts/train_verified_dpo.py --model $PIB --data $O/data/BAC.jsonl \
      --lr 5e-6 --seed $S --precompute-ref --out $O/runs/BAC_s$S > $O/logs/train_BAC_s$S.log 2>&1
  [ -f $O/runs/AAC_s$S/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PIB --data $O/data/AAC.jsonl \
      --lr 1e-4 --seed $S --out $O/runs/AAC_s$S > $O/logs/train_AAC_s$S.log 2>&1
  rm -rf $O/runs/*_s$S/checkpoint-*
  (cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --prompts std --out $O/uo \
      --runs OTV_s$S=$O/runs/OTV_s$S/adapter BAC_s$S=$O/runs/BAC_s$S/adapter AAC_s$S=$O/runs/AAC_s$S/adapter > $O/logs/uo_s$S.log 2>&1)
  for ARM in OTV BAC AAC; do
    N=${ARM}_s$S
    [ -f $O/eval/$N.jsonl ] || { python scripts/evaluate_policies.py --model $PIB --adapter $O/runs/$N/adapter --name otv_$N \
        --splits test,heldout --skip-probe > $O/logs/eval_$N.log 2>&1
      mv results/eval/otv_$N.jsonl $O/eval/$N.jsonl; mv results/eval/otv_$N.summary.json $O/eval/$N.summary.json; }
  done
done
DONE_SEEDS=$(for f in $O/uo/OTV_s*.jsonl; do basename $f .jsonl | sed 's/OTV_s//'; done | sort | tr '\n' ' ')
python scripts/otv_analysis.py --uo $O/uo --seeds $DONE_SEEDS --out $O/analysis.json | tee $O/logs/analysis.log
tar --exclude=__pycache__ -czf $W/otv_results.tgz -C $W otv/uo otv/eval otv/logs otv/analysis.json otv/data/MANIFEST.json \
    $(cd $W && ls -d otv/runs/*/adapter otv/runs/*/run_config.json otv/runs/*/log_history.json 2>/dev/null)
sha256sum $W/otv_results.tgz
echo OTV_DONE
