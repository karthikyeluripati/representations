#!/usr/bin/env bash
# tau2 airline zero-shot transfer evaluation (pre-registered: corrective-training-study/docs/tau2_transfer.md). Eval only.
set -euo pipefail
cd "$(dirname "$0")/.."
W=/workspace; PIB=$W/models/piB; S=$W/synth; O=$W/tau2_eval
export PYTHONHASHSEED=0 HF_HOME=/root/cache/hf VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
unset CTS_TOOL_ANNOTATIONS
mkdir -p $O/logs
source /root/venv/bin/activate
pip install -q loguru docstring-parser deepdiff addict toml python-dotenv
python -c "import pydantic; assert tuple(map(int, pydantic.VERSION.split('.')[:2])) >= (2, 12), pydantic.VERSION"
python scripts/tau2_validate.py > $O/logs/validate.log 2>&1
grep -q "verify       ACa: 39/39  ACl: 39/39" $O/logs/validate.log || { echo "ABORT: tau2 validation differs on pod"; cat $O/logs/validate.log; exit 5; }
RUNS="piB="
for A in SYNOLD SYNIAR SYN64IAR; do for s in 0 1 2; do [ -f $S/runs/${A}_s$s/adapter/adapter_config.json ] && RUNS="$RUNS ${A}_s$s=$S/runs/${A}_s$s/adapter"; done; done
echo "runs: $RUNS"
python scripts/eval_tau2_ac.py --model $PIB --out $O --runs $RUNS > $O/logs/eval.log 2>&1
python scripts/tau2_analysis.py --dir $O --arms SYNOLD SYNIAR SYN64IAR --out $O/analysis.json > $O/logs/analysis.log
tar -czf $W/tau2_eval_slim.tgz -C $W tau2_eval && sha256sum $W/tau2_eval_slim.tgz
echo TAU2_DONE
