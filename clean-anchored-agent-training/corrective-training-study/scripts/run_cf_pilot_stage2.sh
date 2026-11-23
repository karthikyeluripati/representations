#!/usr/bin/env bash
# Stage-2 driver for the matched CF-SFT pilot (docs/cf_sft_pilot_protocol.md sec 5). Calls the FROZEN
# run_cf_pilot_h100.sh unchanged, one job at a time, on one pod. Order: all 6 trainings (+ T2/T1 diagnostics) first,
# then the regression suite (REG=1) for CF_s0-s2 and CTRL_s0. Existing adapters and traces are skipped by the frozen
# script, so stage-1 outputs are never overwritten.
# A failed job is recorded and the driver continues ("report every seed, including failures").
# Contract with scripts/pod_guard.sh: writes its PID to $PIDF; last steps are archive + sha256, then the DRIVER_* marker.
set -uo pipefail
W=/workspace; REPO=$W/clean-anchored-agent-training; O=$W/cf_pilot; LOG=$W/cfp2.log; PIDF=$W/cfp2.pid
ARC=$W/cf_pilot_stage2_${STAMP:?}.tgz
echo $$ > $PIDF
cd $REPO
S=corrective-training-study/scripts/run_cf_pilot_h100.sh
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $LOG; }

# preflight (seconds): reference traces the S4 gate copies; abort before any training if absent
miss=0
for g in "$W/synth/eval/CAW2_s*.jsonl" "$W/synth/uo/CAW2_s*.jsonl" "$W/synth/uo/piB.jsonl" "$W/tau2_eval/CAW2_s*.jsonl" \
         "$O/runs/CF_s0/adapter/adapter_config.json" "$O/runs/CTRL_s0/adapter/adapter_config.json"; do
  compgen -G "$g" > /dev/null || { say "PREFLIGHT missing: $g"; miss=1; }
done
[ $miss = 1 ] && { echo "DRIVER_FAILED preflight" >> $LOG; exit 1; }
say "preflight ok"

status=0
job() {  # ARM SEED REG
  say "START $1_s$2 REG=$3"
  if STAGE=2 ARM=$1 SEED=$2 REG=$3 bash $S > $W/cfp2_$1_s$2_reg$3.log 2>&1; then say "OK $1_s$2 REG=$3"
  else say "FAILED $1_s$2 REG=$3 (rc=$?; tail below)"; tail -5 $W/cfp2_$1_s$2_reg$3.log | tee -a $LOG; status=1; fi
}
for j in "CF 1" "CTRL 1" "ABL 0" "CF 2" "CTRL 2" "ABL 1"; do job $j 0; done
for j in "CF 0" "CF 1" "CF 2" "CTRL 0"; do job $j 1; done

say "archiving"
cd $W
F=$(ls -d cf_pilot/t2test cf_pilot/t1 cf_pilot/reg cf_pilot/logs cf_pilot/data/MANIFEST_render_*.json \
        cf_pilot/runs/*/run_config.json cf_pilot/runs/*/log_history.json cf_pilot/runs/*/timing.json \
        cf_pilot/runs/*/adapter/adapter_config.json cfp2*.log cfp2_driver.sh 2>/dev/null)
(cd cf_pilot/runs && sha256sum */adapter/adapter_model.safetensors) > cf_pilot/ADAPTER_SHA256SUMS
tar czf $ARC $F cf_pilot/ADAPTER_SHA256SUMS && (cd $W && sha256sum $(basename $ARC) > $ARC.sha256) || status=1
say "archive $(cat $ARC.sha256 2>/dev/null)"
[ $status = 0 ] && echo "DRIVER_DONE" >> $LOG || echo "DRIVER_FAILED (see FAILED lines; archive written)" >> $LOG
