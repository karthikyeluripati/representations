#!/usr/bin/env bash
# Wait until a marker line appears in a file, then run a command. Usage: queue_after.sh FILE MARKER CMD...
FILE=$1; MARKER=$2; shift 2
until grep -q "$MARKER" "$FILE" 2>/dev/null; do sleep 60; done
exec "$@"
