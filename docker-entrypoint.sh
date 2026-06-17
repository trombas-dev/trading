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

# Create per-symbol state subdirectories for the 6 active Fibonacci pairs.
for sym in GBPAUD BTCUSD GBPUSD USDCHF XAUUSD NZDUSD; do
    mkdir -p "$STATE/$sym"
done

# Seed breakout_strategy.yaml if not already present
if [ ! -f "$STATE/breakout_strategy.yaml" ]; then
    if [ -f "$DEFAULTS/breakout_strategy.yaml" ]; then
        cp "$DEFAULTS/breakout_strategy.yaml" "$STATE/breakout_strategy.yaml"
        echo "[entrypoint] seeded breakout_strategy.yaml"
    fi
fi

# Create per-symbol state dirs for the 8 breakout instruments.
for sym in XNGUSD XAUUSD BTCUSD US500 US30 XTIUSD JP225 XBRUSD; do
    mkdir -p "$STATE/breakout/$sym"
done

exec "$@"
