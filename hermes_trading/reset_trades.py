"""
reset_trades.py — wipes all trades.jsonl files so stats start fresh.

Usage (run once on Railway via one-off command):
  uv run python -m hermes_trading.reset_trades
"""
from pathlib import Path
import sys

state_dir = Path(__file__).parent.parent / "state"

FIB_SYMBOLS = ["GBPAUD", "BTCUSD", "GBPUSD", "USDCHF", "XAUUSD", "NZDUSD"]
BO_SYMBOLS  = ["US30", "US500", "JP225", "XAUUSD", "XTIUSD", "XBRUSD", "XNGUSD", "BTCUSD"]

deleted = []

for sym in FIB_SYMBOLS:
    f = state_dir / sym / "trades.jsonl"
    if f.exists():
        f.unlink()
        deleted.append(str(f))

for sym in BO_SYMBOLS:
    f = state_dir / "breakout" / sym / "trades.jsonl"
    if f.exists():
        f.unlink()
        deleted.append(str(f))

if deleted:
    print(f"Deleted {len(deleted)} trade files:")
    for p in deleted:
        print(f"  {p}")
else:
    print("No trade files found — already clean.")
