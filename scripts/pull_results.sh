#!/usr/bin/env bash
# Periodically mirror a pod's results/logs into results/pod_sync/<tag>/ on this machine.
# Usage: bash scripts/pull_results.sh HOST PORT TAG [INTERVAL_S]
set -u
HOST=$1; PORT=$2; TAG=$3; INT=${4:-600}
cd "$(dirname "$0")/.."
DEST="results/pod_sync/$TAG"; mkdir -p results/pod_sync
while true; do
  TMP="$DEST.tmp"; rm -rf "$TMP"; mkdir -p "$TMP"
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "root@$HOST" -p "$PORT" \
       'cd /workspace/dare && tar czf - results/decisions_v1_7b results/decisions_v1_3b logs 2>/dev/null' \
       | tar --force-local --no-same-owner -xzf - -C "$TMP" 2>/dev/null; then
    # copy over (rows are append-only; renames are unreliable under OneDrive on Windows)
    mkdir -p "$DEST" && cp -rf "$TMP/." "$DEST/" && rm -rf "$TMP"
    echo "$(date -u +%H:%M) synced: 7b=$(cat $DEST/results/decisions_v1_7b/decisions*.jsonl 2>/dev/null | wc -l) 3b=$(cat $DEST/results/decisions_v1_3b/decisions*.jsonl 2>/dev/null | wc -l)"
  else
    echo "$(date -u +%H:%M) sync FAILED (keeping previous copy)"
  fi
  sleep "$INT"
done
