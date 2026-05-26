"""
MTF Fibonacci Retracement strategy — modular components.

Build order:
  Step 1  swing.py       — N-candle swing high/low detection
  Step 2  structure.py   — BOS / CHoCH market structure
  Step 3  fibonacci.py   — retracement levels + extensions
  Step 4  signal.py      — zone detection on signal timeframe
  Step 5  entry.py       — bounce confirmation on entry timeframe
  Step 6  exits.py       — virtual SL + TP1/TP2 management
"""
