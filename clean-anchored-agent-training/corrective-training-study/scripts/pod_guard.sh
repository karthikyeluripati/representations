#!/usr/bin/env bash
# Budget guard for ONE RunPod pod (replaces the inline stage-1 guard).
# The stage-1 guard had two bugs:
#   (1) its done-check was `ssh ... 'pgrep -f cfp1_driver.sh'`. The remote shell's own command line contains that
#       string, so pgrep always matched itself and the guard never saw completion.
#   (2) it stopped the pod only on timeout, never on completion.
# This guard:
#   * detects completion from a marker line in the driver log (DRIVER_DONE / DRIVER_FAILED), plus a PID-file liveness
#     check (kill -0). No process-name matching.
#   * on completion: fetches the archive the driver wrote, verifies its sha256 locally, stops the pod, and confirms
#     desiredStatus=EXITED and that the account spend rate dropped to storage only.
#   * stops the pod on: completion, driver died without a marker, time cap, or MAX_SSH_FAIL consecutive SSH failures.
#     A stop never deletes anything: outputs stay on the network volume.
# Required env:
#   POD_ID T0(epoch s) MAX_SECONDS SSH_HOST SSH_PORT
#   R_LOG(remote driver log) R_PID(remote pid file) R_ARCHIVE(remote .tgz; driver also writes .tgz.sha256) LOCAL_DIR
# Optional env:
#   POLL=60  MAX_SSH_FAIL=15  RUNPOD_API_KEY (real mode)  GUARD_TEST=1 (CPU test mode: runs "remote" commands locally;
#   stop/status use the file $FAKE_POD)
set -uo pipefail
: "${POD_ID:?}" "${T0:?}" "${MAX_SECONDS:?}" "${R_LOG:?}" "${R_PID:?}" "${R_ARCHIVE:?}" "${LOCAL_DIR:?}"
POLL=${POLL:-60}; MAX_SSH_FAIL=${MAX_SSH_FAIL:-15}; TEST=${GUARD_TEST:-0}
log() { echo "[guard $(date -u +%H:%M:%S) +$(( $(date -u +%s) - T0 ))s] $*"; }

remote() {  # run a command on the pod; prints stdout; returns 255 on transport failure
  if [ "$TEST" = 1 ]; then [ -f "${FAKE_SSH_DOWN:-/nonexistent}" ] && return 255; bash -c "$1"
  else ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -p "$SSH_PORT" "root@$SSH_HOST" "$1"; fi
}
fetch() {   # remote path -> local dir
  if [ "$TEST" = 1 ]; then cp "$1" "$2/"
  else scp -q -o StrictHostKeyChecking=no -o ConnectTimeout=15 -P "$SSH_PORT" "root@$SSH_HOST:$1" "$2/"; fi
}
gql() {
  curl -s -m 30 -X POST https://api.runpod.io/graphql -H "Content-Type: application/json" \
       -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"$1\"}"
}
stop_pod() {
  if [ "$TEST" = 1 ]; then echo EXITED > "$FAKE_POD"; echo "{\"podStop\":\"$POD_ID\"}"
  else gql "mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }"; fi
}
pod_status() {  # prints "desiredStatus spendPerHr"
  if [ "$TEST" = 1 ]; then s=$(cat "$FAKE_POD"); [ "$s" = EXITED ] && echo "EXITED 0.015" || echo "$s 4.39"
  else r=$(gql "query { pod(input: {podId: \\\"$POD_ID\\\"}) { desiredStatus } myself { currentSpendPerHr } }")
       echo "$(echo "$r" | grep -o '"desiredStatus":"[A-Z]*"' | cut -d'"' -f4) $(echo "$r" | grep -o '"currentSpendPerHr":[0-9.]*' | cut -d: -f2)"; fi
}

PROBE="if grep -qE '^DRIVER_(DONE|FAILED)' '$R_LOG' 2>/dev/null; then grep -E '^DRIVER_(DONE|FAILED)' '$R_LOG' | tail -1;
elif [ -s '$R_PID' ] && kill -0 \$(cat '$R_PID') 2>/dev/null; then echo RUNNING; else echo DEAD; fi"

finish() {  # $1 = reason; fetch (only when the driver finished), stop, verify billing stopped
  local reason=$1 archive_ok=NO
  if [ "$reason" = DONE ]; then
    mkdir -p "$LOCAL_DIR"
    for i in 1 2 3; do
      if fetch "$R_ARCHIVE" "$LOCAL_DIR" && fetch "$R_ARCHIVE.sha256" "$LOCAL_DIR"; then
        want=$(cut -d' ' -f1 "$LOCAL_DIR/$(basename "$R_ARCHIVE").sha256")
        got=$(sha256sum "$LOCAL_DIR/$(basename "$R_ARCHIVE")" | cut -d' ' -f1)
        [ -n "$want" ] && [ "$want" = "$got" ] && { archive_ok=YES; break; }
        log "archive sha mismatch (want $want got $got), retry $i"
      else log "archive fetch failed, retry $i"; fi
      sleep 5
    done
  fi
  log "STOPPING pod $POD_ID (reason=$reason, archive_verified=$archive_ok)"; stop_pod; echo
  for i in $(seq 1 10); do
    read -r st spend <<<"$(pod_status)"
    if [ "$st" = EXITED ] && awk "BEGIN{exit !(${spend:-9} < 0.5)}"; then
      log "VERIFIED stopped: desiredStatus=$st spendPerHr=$spend"; echo "GUARD_RESULT reason=$reason archive_verified=$archive_ok stopped=YES"; return 0; fi
    log "not yet stopped ($st $spend); re-issuing stop"; stop_pod >/dev/null; sleep 10
  done
  echo "GUARD_RESULT reason=$reason archive_verified=$archive_ok stopped=UNCONFIRMED"; return 1
}

fails=0
while true; do
  el=$(( $(date -u +%s) - T0 ))
  if [ "$el" -ge "$MAX_SECONDS" ]; then finish TIMECAP; exit $?; fi
  out=$(remote "$PROBE"); rc=$?
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    fails=$((fails + 1)); log "ssh probe failed ($fails/$MAX_SSH_FAIL)"
    [ $fails -ge "$MAX_SSH_FAIL" ] && { finish SSH_LOST; exit $?; }
  else
    fails=0
    case "$out" in
      DRIVER_DONE*)   log "$out"; finish DONE; exit $? ;;
      DRIVER_FAILED*) log "$out"; finish DONE; exit $? ;;   # still fetch the archive of whatever completed
      DEAD)           log "driver not running and no marker"; finish DRIVER_DEAD; exit $? ;;
      RUNNING)        : ;;
      *)              log "unexpected probe output: $out" ;;
    esac
  fi
  sleep "$POLL"
done
