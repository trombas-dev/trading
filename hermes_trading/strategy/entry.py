"""
entry.py — Bounce confirmation on the entry timeframe (5 M).

Pipeline position
-----------------
  1H  (BIAS)    find_swings -> find_structure -> fibs_from_last_choch
  15M (SIGNAL)  find_swings -> find_structure -> find_signals
                     |
               SignalResult (active 15M signals)
                     |
                     v
  5M  (ENTRY)   find_swings -> find_structure -> find_entries  <- this module

How it works
------------
Each time a 15M signal fires the entry module opens a ``timeout_bars``-wide
window on the 5M chart.  Inside that window it watches for a 5M BOS or CHoCH
in the direction of the 1H bias.  The first such event becomes the entry bar.

Only ONE entry is allowed per 15M signal.  If a new 15M signal fires while a
window is still open the window is refreshed (timeout reset) and the new
signal replaces the old one.

Entry price
-----------
The entry price is the CLOSE of the confirming 5M bar.  In live trading you
would use the OPEN of the next bar; the close is used here to avoid look-ahead
in simulation.

Preliminary stop-loss
---------------------
Computed in this module as a placeholder until Step 6 (virtual SL):
  bullish: SL = min(entry_bar_low, fib_786_price) – 1 tick
  bearish: SL = max(entry_bar_high, fib_786_price) + 1 tick
``1 tick'' = 0.01 % of price (overridden in Step 6 with close-based SL).

R:R preview
-----------
  risk    = abs(entry_price - stop_loss)
  reward1 = abs(tp1_price   - entry_price)
  rr1     = reward1 / risk

Variables for learning
----------------------
  timeout_bars : int  default 4
                 How many 5M bars the entry window stays open after a 15M
                 signal.  4 bars = 20 minutes.
                 Valid: 2, 4, 6, 8, 12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from hermes_trading.strategy.fibonacci import FibLevel, FibResult
from hermes_trading.strategy.signal import SignalEvent, SignalResult
from hermes_trading.strategy.structure import StructureEvent, StructureResult

# ── Public types ─────────────────────────────────────────────────────────────

Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class EntrySignal:
    """One confirmed entry on the 5M timeframe."""

    index:         int              # bar in the 5M DataFrame
    timestamp:     pd.Timestamp
    direction:     Direction        # 1H bias (trade direction)
    entry_price:   float            # close of the confirming 5M bar
    stop_loss:     float            # preliminary SL (overridden in Step 6)
    risk:          float            # entry_price - stop_loss (always positive)
    tp1:           float            # TP1 price (127.2 % extension)
    tp2:           float            # TP2 price (161.8 % extension)
    rr1:           float            # reward/risk to TP1
    nearest_fib:   FibLevel         # closest fib level to entry price
    signal_15m:    SignalEvent      # the 15M signal that opened this window
    confirmed_by:  StructureEvent   # the 5M structure event that fired

    def __repr__(self) -> str:
        return (
            f"EntrySignal({self.direction:7s}  "
            f"entry={self.entry_price:>12.4f}  "
            f"sl={self.stop_loss:>12.4f}  "
            f"tp1={self.tp1:>12.4f}  "
            f"R:R={self.rr1:.2f}  "
            f"at={str(self.timestamp)[:16]}  "
            f"[5M {self.confirmed_by.kind.upper()}])"
        )


@dataclass
class EntryResult:
    """All entry signals from a single pass over the 5M timeframe."""

    entries:  list[EntrySignal] = field(default_factory=list)
    fib:      FibResult | None  = None
    bias:     Direction         = "bullish"
    timeout_bars: int           = 4

    # ── Accessors ─────────────────────────────────────────────────────────────

    def entries_by(self, bar_index: int) -> list[EntrySignal]:
        return [e for e in self.entries if e.index <= bar_index]

    def last_entry(self, as_of: int | None = None) -> EntrySignal | None:
        pool = self.entries_by(as_of) if as_of is not None else self.entries
        return pool[-1] if pool else None

    @property
    def avg_rr1(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e.rr1 for e in self.entries) / len(self.entries)

    def summary(self) -> str:
        return (
            f"EntryResult  "
            f"bias={self.bias}  "
            f"entries={len(self.entries)}  "
            f"avg_R:R1={self.avg_rr1:.2f}  "
            f"timeout={self.timeout_bars} bars"
        )


# ── Core detector ─────────────────────────────────────────────────────────────

def find_entries(
    df: pd.DataFrame,
    sig_res: SignalResult,
    struct_res: StructureResult,
    fib: FibResult,
    close_col:    str  = "close",
    high_col:     str  = "high",
    low_col:      str  = "low",
    time_col:     str | None = None,
    timeout_bars: int  = 4,
) -> EntryResult:
    """
    Detect entry confirmations on the 5M timeframe.

    Parameters
    ----------
    df           : 5M OHLCV DataFrame.
    sig_res      : SignalResult from find_signals() on the 15M chart.
    struct_res   : StructureResult from find_structure() on the same df (5M).
    fib          : FibResult from the 1H bias layer.
    close_col    : Close price column.
    high_col     : High price column.
    low_col      : Low price column.
    time_col     : Timestamp column; falls back to df.index.
    timeout_bars : 5M bars the entry window stays open after a 15M signal.

    Returns
    -------
    EntryResult with all EntrySignal objects in chronological order.
    """
    closes = df[close_col].to_numpy(dtype=float)
    highs  = df[high_col].to_numpy(dtype=float)
    lows   = df[low_col].to_numpy(dtype=float)
    n_bars = len(closes)

    if time_col:
        timestamps = pd.to_datetime(df[time_col])
    else:
        timestamps = (
            df.index
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.RangeIndex(n_bars)
        )

    bias: Direction = sig_res.bias

    # ── Pre-build 5M struct lookup: bar -> events in bias direction ───────────
    struct_by_bar: dict[int, list[StructureEvent]] = {}
    for e in struct_res.events:
        if e.direction == bias:
            struct_by_bar.setdefault(e.index, []).append(e)

    # ── Sort 15M signals by timestamp for O(n) scan ───────────────────────────
    sig_list = sorted(sig_res.signals, key=lambda s: s.timestamp)
    sig_ptr  = 0

    # Pre-compute fib levels for SL / TP
    fib_786 = fib.level_at(0.786)
    fib_100 = fib.level_at(1.0)    # swing end (100 % retracement)
    tp1_price = fib.tp1.price if fib.tp1 else 0.0
    tp2_price = fib.tp2.price if fib.tp2 else 0.0
    tick      = 0.0001             # 0.01 % placeholder tick size

    active_signal:  SignalEvent | None = None
    active_expires: int                = 0    # 5M bar when window closes

    entries: list[EntrySignal] = []

    for i in range(n_bars):
        ts = timestamps[i]

        # ── Activate 15M signals whose timestamp <= this 5M bar ───────────────
        while sig_ptr < len(sig_list) and sig_list[sig_ptr].timestamp <= ts:
            active_signal  = sig_list[sig_ptr]
            active_expires = i + timeout_bars
            sig_ptr += 1

        # ── Expire the active window ──────────────────────────────────────────
        if active_signal is not None and i > active_expires:
            active_signal = None

        if active_signal is None:
            continue

        # ── Look for 5M structure confirmation in bias direction ──────────────
        confirming = struct_by_bar.get(i, [])
        if not confirming:
            continue

        struct_evt = confirming[0]
        close      = closes[i]
        high       = highs[i]
        low        = lows[i]

        # ── Preliminary stop-loss ─────────────────────────────────────────────
        if bias == "bullish":
            # SL just below the 78.6 % level (or bar low, whichever is lower)
            sl_ref = fib_786.price if fib_786 else (fib_100.price if fib_100 else low)
            sl     = min(low, sl_ref) * (1.0 - tick)
        else:
            sl_ref = fib_786.price if fib_786 else (fib_100.price if fib_100 else high)
            sl     = max(high, sl_ref) * (1.0 + tick)

        risk = abs(close - sl)
        if risk < 1e-8:
            continue    # degenerate bar — skip

        rr1 = abs(tp1_price - close) / risk if tp1_price else 0.0

        entries.append(EntrySignal(
            index        = i,
            timestamp    = timestamps[i],
            direction    = bias,
            entry_price  = close,
            stop_loss    = sl,
            risk         = risk,
            tp1          = tp1_price,
            tp2          = tp2_price,
            rr1          = rr1,
            nearest_fib  = fib.nearest_level(close),
            signal_15m   = active_signal,
            confirmed_by = struct_evt,
        ))

        # Consume: one entry per 15M signal
        active_signal = None

    return EntryResult(
        entries      = entries,
        fib          = fib,
        bias         = bias,
        timeout_bars = timeout_bars,
    )
