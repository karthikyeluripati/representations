#!/usr/bin/env bash
# Semantic-counterfactual controlled evaluation (docs/semantic_counterfactual_protocol.md, edit log 1). Eval only.
set -euo pipefail
cd /workspace/clean-anchored-agent-training/corrective-training-study
W=/workspace; PIB=$W/models/piB; R=$W/synth/runs; O=$W/semcf_eval; mkdir -p $O/logs /root/cache
export PYTHONHASHSEED=0 PIP_CACHE_DIR=/root/cache/pip HF_HOME=/root/cache/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q "vllm==0.11.0" "transformers==4.57.3" "peft==0.17.1" "accelerate==1.10.1" pytest > $O/logs/pip.log 2>&1
pip install -q --no-deps "bfcl-eval==2026.3.23" >> $O/logs/pip.log 2>&1
(cd $PIB && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: piB checksum mismatch"; exit 3; }
python -m pytest -q tests/test_semantic_cf.py tests/test_semantic_cf_eval.py > $O/logs/tests.log 2>&1 || { tail -20 $O/logs/tests.log; exit 4; }
python -c "import sys; sys.path.insert(0,'src'); from verification.semantic_cf import specs; import hashlib,json; h=hashlib.sha256(json.dumps(specs('test'),sort_keys=True).encode()).hexdigest(); print(h); assert h.startswith('a591d07a')" > $O/logs/specs_hash.log
python scripts/eval_semantic_cf.py --model $PIB --out $O --runs piB= piB_prompt= \
  SYN64_s0=$R/SYN64IAR_s0/adapter SYN64_s1=$R/SYN64IAR_s1/adapter SYN64_s2=$R/SYN64IAR_s2/adapter \
  CAW_s0=$R/CAW64_s0/adapter CAW_s1=$R/CAW64_s1/adapter CAW2_s0=$R/CAW2_s0/adapter CAW2_s1=$R/CAW2_s1/adapter > $O/logs/eval.log 2>&1
python scripts/semantic_cf_analysis.py --dir $O --out $O/analysis.json --md $O/analysis.md > $O/logs/analysis.log 2>&1
echo SEMCF_DONE
