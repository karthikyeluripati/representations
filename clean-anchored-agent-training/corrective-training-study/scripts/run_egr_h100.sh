#!/usr/bin/env bash
# Evidence-grounded recovery study (pre-registered: docs/method_egr.md). Resumable; SEEDS default "0".
set -euo pipefail
cd "$(dirname "$0")/.."
STUDY=$(pwd); PROJ=$(cd .. && pwd)
W=/workspace; R=$W/replication; PI0=$W/models/pi0; PIB=$W/models/piB; O=$W/egr; SEEDS=${SEEDS:-"0"}
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
declare -A CL=( [M]="TravelAPI,TradingBot,TwitterAPI,TicketAPI" [T]="MessageAPI,TradingBot,TwitterAPI,TicketAPI" )
declare -A DV=( [M]="otv_v2" [T]="otv_v2T" )
for F in M T; do
  python scripts/build_otv_pairs.py --version ${DV[$F]} --train-classes ${CL[$F]} --out-name pairs_plain.jsonl > $O/logs/build_${F}_plain.log 2>&1
  python scripts/build_otv_pairs.py --version ${DV[$F]} --train-classes ${CL[$F]} --out-name pairs_evidence.jsonl --evidence > $O/logs/build_${F}_ev.log 2>&1
done
python - <<EOF
import json
for v, n in (("otv_v2", 544), ("otv_v2T", 352)):
    m = json.load(open(f"$STUDY/data/{v}/MANIFEST.json"))
    assert m["pairs_plain.jsonl"]["n_pairs"] == n and m["pairs_evidence.jsonl"]["n_pairs"] == n, (v, m)
print("pairs ok")
EOF
CTS_TOKENIZER=$PI0 python -m pytest -q tests > $O/logs/tests_study.log 2>&1 || { tail -20 $O/logs/tests_study.log; exit 4; }
for F in M T; do
  [ -f $O/data/MANIFEST$F.json ] || python scripts/render_egr_arms.py --tokenizer $PIB --b-dpo $R/data/cts_rep/dpo.jsonl \
      --data-dir $STUDY/data/${DV[$F]} --tag $F --out $O/data > $O/logs/render_$F.log 2>&1
done
[ -f $O/uo/piB.jsonl ] || cp $W/otv/uo/piB.jsonl $W/otv/uo/piB.summary.json $O/uo/

for S in $SEEDS; do
  RUNS=""
  for F in M T; do for ARM in AAC2 EGR; do
    N=${ARM}${F}_s$S
    [ -f $O/runs/$N/adapter/adapter_config.json ] || python scripts/train_balanced_sft.py --model $PIB --data $O/data/${ARM}${F}.jsonl \
        --lr 1e-4 --seed $S --out $O/runs/$N > $O/logs/train_$N.log 2>&1
    rm -rf $O/runs/$N/checkpoint-*
    RUNS="$RUNS $N=$O/runs/$N/adapter"
  done; done
  (cd $PROJ && python scripts/eval_uncertain_outcome.py --model $PIB --prompts std --out $O/uo --runs $RUNS > $O/logs/uo_s$S.log 2>&1)
  for F in M T; do for ARM in AAC2 EGR; do
    N=${ARM}${F}_s$S
    [ -f $O/eval/$N.jsonl ] || { python scripts/evaluate_policies.py --model $PIB --adapter $O/runs/$N/adapter --name egr_$N \
        --splits test,heldout --skip-probe > $O/logs/eval_$N.log 2>&1
      mv results/eval/egr_$N.jsonl $O/eval/$N.jsonl; mv results/eval/egr_$N.summary.json $O/eval/$N.summary.json; }
  done; done
done
DS=$(for f in $O/uo/EGRM_s*.jsonl; do basename $f .jsonl | sed 's/EGRM_s//'; done | sort | tr '\n' ' ')
for F in M T; do
  python scripts/otv_analysis.py --uo $O/uo --arms EGR$F AAC2$F --method EGR$F --train-classes ${CL[$F]} --seeds $DS \
      --out $O/analysis_$F.json > $O/logs/analysis_$F.log
done
python scripts/pool_folds.py --fold M $O/analysis_M.json EGRM AAC2M --fold T $O/analysis_T.json EGRT AAC2T --out $O/pooled.json | tee $O/logs/pooled.log
tar --exclude=__pycache__ -czf $W/egr_slim.tgz -C $W egr/uo egr/eval egr/logs egr/analysis_M.json egr/analysis_T.json egr/pooled.json \
    $(cd $W && ls egr/data/MANIFEST*.json egr/runs/*/run_config.json egr/runs/*/log_history.json 2>/dev/null)
sha256sum $W/egr_slim.tgz
echo EGR_DONE
