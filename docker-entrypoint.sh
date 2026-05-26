#!/bin/sh
set -e

# Seed the persistent volume on first boot.
# The image ships defaults in /app/state-defaults/.
# Copy each file to /app/state/ only if it doesn't already exist.
DEFAULTS=/app/state-defaults
STATE=/app/state

mkdir -p "$STATE/history"

for f in "$DEFAULTS"/*.yaml "$DEFAULTS"/*.jsonl "$DEFAULTS"/*.json; do
    [ -f "$f" ] || continue
    fname="$(basename "$f")"
    if [ ! -f "$STATE/$fname" ]; then
        cp "$f" "$STATE/$fname"
        echo "[entrypoint] seeded $fname"
    fi
done

# Create per-symbol state subdirectories for the 6 active pairs.
# The loop writes heartbeat.json and trades.jsonl into these dirs.
for sym in GBPAUD BTCUSD GBPUSD USDCHF XAUUSD NZDUSD; do
    mkdir -p "$STATE/$sym"
done

exec "$@"
