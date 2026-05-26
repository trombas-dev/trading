"""
structure.py — BOS / CHoCH market structure classifier.

Definitions
-----------
BOS  (Break of Structure)
    Close breaks a confirmed swing level IN THE DIRECTION of the current
    bias → trend continuation.

CHoCH (Change of Character)
    Close breaks a confirmed swing level AGAINST the current bias
    → trend reversal signal.

Event table
-----------
  Current bias | Break direction | Event
  -------------|-----------------|----------------------
  bullish      | above swing high| Bullish BOS
  bullish      | below swing low | Bearish CHoCH  ← reversal
  bearish      | below swing low | Bearish BOS
  bearish      | above swing high| Bullish CHoCH  ← reversal
  neutral      | above swing high| Bullish BOS    (first event, establishes bias)
  neutral      | below swing low | Bearish BOS    (first event, establishes bias)

Close-price rule
----------------
ALL breaks are tested against CANDLE CLOSE, never intra-bar high/low.
Wicks that pierce a level but close back inside do NOT trigger an event.

No-lookahead guarantee
-----------------------
At bar i the detector only consumes swing points whose
``confirmed_at < i`` (confirmed on a strictly earlier bar).
Safe to call inside any backtesting loop.

Deduplication
-------------
Once a swing level triggers an event it is "used up".  The next event
on the same side (high or low) requires a NEW confirmed swing to be broken.
This prevents the same $X level from re-firing on every subsequent bar.

Variables for learning
----------------------
  n_candles   : inherited from SwingResult — controls swing sensitivity.
                Changing n_candles changes both swing detection AND structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from hermes_trading.strategy.swing import SwingPoint, SwingResult

# ── Public types ─────────────────────────────────────────────────────────────

EventKind = Literal["bos", "choch"]
Direction = Literal["bullish", "bearish"]
Bias      = Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True)
class StructureEvent:
    """One confirmed BOS or CHoCH event."""

    index:        int               # bar where the close crossed the level
    timestamp:    pd.Timestamp      # candle open-time at `index`
    kind:         EventKind         # "bos" or "choch"
    direction:    Direction         # "bullish" or "bearish"
    level:        float             # price of the broken swing
    close:        float             # close that triggered the break
    broken_swing: SwingPoint        # the SwingPoint that was broken

    def __repr__(self) -> str:
        return (
            f"StructureEvent({self.kind.upper():5s}  "
            f"{self.direction:7s}  "
            f"level={self.level:>12.4f}  "
            f"close={self.close:>12.4f}  "
            f"at={str(self.timestamp)[:16]})"
        )


@dataclass
class StructureResult:
    """All structure events found in a single pass, with convenience accessors."""

    events: list[StructureEvent] = field(default_factory=list)

    # ── Filters ──────────────────────────────────────────────────────────────

    @property
    def bos(self) -> list[StructureEvent]:
        return [e for e in self.events if e.kind == "bos"]

    @property
    def choch(self) -> list[StructureEvent]:
        return [e for e in self.events if e.kind == "choch"]

    @property
    def bullish_events(self) -> list[StructureEvent]:
        return [e for e in self.events if e.direction == "bullish"]

    @property
    def bearish_events(self) -> list[StructureEvent]:
        return [e for e in self.events if e.direction == "bearish"]

    # ── Time-aware accessors (no-lookahead safe) ──────────────────────────────

    def events_by(self, bar_index: int) -> list[StructureEvent]:
        """All events that occurred on or before `bar_index`."""
        return [e for e in self.events if e.index <= bar_index]

    def last_event(self, as_of: int | None = None) -> StructureEvent | None:
        pool = self.events_by(as_of) if as_of is not None else self.events
        return pool[-1] if pool else None

    def last_bos(self, as_of: int | None = None) -> StructureEvent | None:
        pool = self.events_by(as_of) if as_of is not None else self.events
        bos = [e for e in pool if e.kind == "bos"]
        return bos[-1] if bos else None

    def last_choch(self, as_of: int | None = None) -> StructureEvent | None:
        pool = self.events_by(as_of) if as_of is not None else self.events
        ch = [e for e in pool if e.kind == "choch"]
        return ch[-1] if ch else None

    def bias(self, as_of: int | None = None) -> Bias:
        """
        Market bias at `as_of` bar.

        Returns "neutral" until the first event occurs, then reflects the
        direction of the last event (BOS or CHoCH) seen up to that point.
        """
        evt = self.last_event(as_of)
        return evt.direction if evt is not None else "neutral"

    def last_n_events(self, n: int, as_of: int | None = None) -> list[StructureEvent]:
        pool = self.events_by(as_of) if as_of is not None else self.events
        return pool[-n:]

    def summary(self) -> str:
        return (
            f"StructureResult  "
            f"total={len(self.events)}  "
            f"bos={len(self.bos)}  "
            f"choch={len(self.choch)}  "
            f"bullish={len(self.bullish_events)}  "
            f"bearish={len(self.bearish_events)}"
        )


# ── Core detector ─────────────────────────────────────────────────────────────

def find_structure(
    df: pd.DataFrame,
    swings: SwingResult,
    close_col: str = "close",
    time_col:  str | None = None,
) -> StructureResult:
    """
    Detect BOS and CHoCH events from close-price breaks of swing levels.

    Parameters
    ----------
    df        : DataFrame with at least `close_col` (same index as passed to
                find_swings).
    swings    : SwingResult produced by find_swings() on the same df.
    close_col : Column name for candle close prices.
    time_col  : If set, read timestamps from this column; else use df.index.

    Returns
    -------
    StructureResult with all StructureEvent objects in chronological order.

    Algorithm (O(n) time)
    ---------------------
    Two forward-only pointers scan the confirmed-swing lists in step with the
    main bar loop.  Each pointer advances only when a swing's confirmed_at
    is strictly less than the current bar index, ensuring no lookahead.
    Once a swing fires an event it is marked "used" and cannot fire again,
    preventing stale-level re-triggers.
    """
    closes = df[close_col].to_numpy(dtype=float)
    n_bars = len(closes)

    if time_col:
        timestamps = pd.to_datetime(df[time_col])
    else:
        timestamps = (
            df.index
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.RangeIndex(n_bars)
        )

    all_highs: list[SwingPoint] = swings.highs   # already sorted by index
    all_lows:  list[SwingPoint] = swings.lows

    events: list[StructureEvent] = []

    bias: Bias = "neutral"
    last_used_high: SwingPoint | None = None
    last_used_low:  SwingPoint | None = None

    # Forward pointers: h_ptr / l_ptr = first swing NOT YET confirmed at bar i
    h_ptr = 0
    l_ptr = 0

    for i in range(n_bars):
        close = closes[i]

        # Advance pointers to include all swings confirmed strictly before bar i
        # confirmed_at < i  ↔  the confirmation bar has already closed
        while h_ptr < len(all_highs) and all_highs[h_ptr].confirmed_at < i:
            h_ptr += 1
        while l_ptr < len(all_lows) and all_lows[l_ptr].confirmed_at < i:
            l_ptr += 1

        # Most recent confirmed swing on each side
        cur_high: SwingPoint | None = all_highs[h_ptr - 1] if h_ptr > 0 else None
        cur_low:  SwingPoint | None = all_lows[l_ptr - 1]  if l_ptr > 0 else None

        if cur_high is None or cur_low is None:
            continue

        # ── Upside break: close > last confirmed swing high ───────────────────
        if close > cur_high.price and cur_high is not last_used_high:
            kind: EventKind = "bos" if bias in ("bullish", "neutral") else "choch"
            events.append(StructureEvent(
                index=i,
                timestamp=timestamps[i],
                kind=kind,
                direction="bullish",
                level=cur_high.price,
                close=close,
                broken_swing=cur_high,
            ))
            bias = "bullish"
            last_used_high = cur_high

        # ── Downside break: close < last confirmed swing low ──────────────────
        elif close < cur_low.price and cur_low is not last_used_low:
            kind = "bos" if bias in ("bearish", "neutral") else "choch"
            events.append(StructureEvent(
                index=i,
                timestamp=timestamps[i],
                kind=kind,
                direction="bearish",
                level=cur_low.price,
                close=close,
                broken_swing=cur_low,
            ))
            bias = "bearish"
            last_used_low = cur_low

    return StructureResult(events=events)
