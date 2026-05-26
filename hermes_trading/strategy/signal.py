"""
signal.py — Zone detection on the signal timeframe (15 M).

Pipeline position
-----------------
  1H  (BIAS)    find_swings -> find_structure -> fibs_from_last_choch
                     |
                fib + bias (Direction)
                     |
                     v
  15M (SIGNAL)  find_swings -> find_structure -> find_signals   <- this module
                     |
                SignalResult
                     |
                     v
  5M  (ENTRY)   entry.py — bounce confirmation  (Step 5)

A signal fires when BOTH of the following are true at the same 15M bar:
  1. The candle CLOSE sits inside the Fibonacci zone derived from the 1H swing.
  2. A 15M BOS or CHoCH in the direction of the 1H bias fires at that bar.

If ``require_struct=False`` condition (2) is dropped and zone entry alone fires
the signal — useful for ranging markets or as a looser filter.

Zone kinds (overlap is possible)
---------------------------------
  golden  0.382 – 0.618   Primary entry band.
  deep    0.618 – 0.786   High-conviction / late entries.
  full    0.382 – 0.786   Both bands combined (zone_lo=0.382, zone_hi=0.786).

Event kinds
-----------
  "zone_entry"   Price first closes into the zone (context, not trade signal).
  "signal"       Trade signal: in zone AND 15M structure in bias direction.
  "zone_exit"    Price closes outside the zone (zone invalidated).

Variables for learning
----------------------
  zone_lo        : float  default 0.382
                   Lower Fibonacci bound of the entry zone.
                   Allowed: 0.236, 0.382, 0.500
  zone_hi        : float  default 0.786
                   Upper Fibonacci bound of the entry zone.
                   Allowed: 0.618, 0.705, 0.786
  require_struct : bool   default True
                   Require a 15M structure event inside the zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from hermes_trading.strategy.fibonacci import FibLevel, FibResult
from hermes_trading.strategy.structure import StructureEvent, StructureResult
from hermes_trading.strategy.swing import SwingResult

# ── Public types ─────────────────────────────────────────────────────────────

Direction  = Literal["bullish", "bearish"]
SignalKind = Literal["zone_entry", "signal", "zone_exit"]


@dataclass(frozen=True)
class SignalEvent:
    """One signal (or context) event on the signal timeframe."""

    index:        int                       # bar in the signal-TF DataFrame
    timestamp:    pd.Timestamp
    kind:         SignalKind
    bias:         Direction                 # 1H bias that governs this signal
    price:        float                     # close at this bar
    nearest_fib:  FibLevel                 # closest fib level to price
    in_golden:    bool                      # inside 0.382–0.618
    in_deep:      bool                      # inside 0.618–0.786
    struct_event: StructureEvent | None = None  # confirming 15M structure event

    def __repr__(self) -> str:
        zone = "GOLDEN" if self.in_golden else ("DEEP" if self.in_deep else "—")
        struct = f"  [{self.struct_event.kind.upper()} {self.struct_event.direction}]" \
                 if self.struct_event else ""
        return (
            f"SignalEvent({self.kind:<12s}  "
            f"{self.bias:7s}  "
            f"price={self.price:>12.4f}  "
            f"zone={zone:<6s}  "
            f"at={str(self.timestamp)[:16]}"
            f"{struct})"
        )


@dataclass
class SignalResult:
    """All signal events from a single pass over the signal timeframe."""

    events:   list[SignalEvent] = field(default_factory=list)
    fib:      FibResult   | None = None
    bias:     Direction          = "bullish"
    zone_lo:  float              = 0.382
    zone_hi:  float              = 0.786

    # ── Filters ───────────────────────────────────────────────────────────────

    @property
    def signals(self) -> list[SignalEvent]:
        """Trade-actionable events only (kind == 'signal')."""
        return [e for e in self.events if e.kind == "signal"]

    @property
    def zone_entries(self) -> list[SignalEvent]:
        return [e for e in self.events if e.kind == "zone_entry"]

    @property
    def zone_exits(self) -> list[SignalEvent]:
        return [e for e in self.events if e.kind == "zone_exit"]

    # ── Time-aware accessors ──────────────────────────────────────────────────

    def events_by(self, bar_index: int) -> list[SignalEvent]:
        """All events at or before `bar_index`."""
        return [e for e in self.events if e.index <= bar_index]

    def last_signal(self, as_of: int | None = None) -> SignalEvent | None:
        pool = self.events_by(as_of) if as_of is not None else self.events
        sigs = [e for e in pool if e.kind == "signal"]
        return sigs[-1] if sigs else None

    def last_event(self, as_of: int | None = None) -> SignalEvent | None:
        pool = self.events_by(as_of) if as_of is not None else self.events
        return pool[-1] if pool else None

    def currently_in_zone(self, bar_index: int) -> bool:
        """True if the last zone event before bar_index is zone_entry."""
        zone_events = [
            e for e in self.events
            if e.index <= bar_index and e.kind in ("zone_entry", "zone_exit")
        ]
        if not zone_events:
            return False
        return zone_events[-1].kind == "zone_entry"

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        return (
            f"SignalResult  "
            f"bias={self.bias}  "
            f"zone={self.zone_lo:.3f}-{self.zone_hi:.3f}  "
            f"signals={len(self.signals)}  "
            f"entries={len(self.zone_entries)}  "
            f"exits={len(self.zone_exits)}"
        )


# ── Core detector ─────────────────────────────────────────────────────────────

def find_signals(
    df: pd.DataFrame,
    fib: FibResult,
    bias: Direction,
    struct_res: StructureResult,
    close_col:      str   = "close",
    time_col:       str | None = None,
    zone_lo:        float = 0.382,
    zone_hi:        float = 0.786,
    require_struct: bool  = True,
) -> SignalResult:
    """
    Detect trade signals on the signal timeframe (15M).

    Parameters
    ----------
    df             : Signal-timeframe DataFrame (e.g. 15M OHLCV).
    fib            : FibResult computed on the BIAS timeframe (1H).
                     Provides price levels — no index alignment needed.
    bias           : Direction from the 1H structure (bullish / bearish).
    struct_res     : StructureResult from find_structure() on the same df.
                     Provides 15M BOS/CHoCH events for confirmation.
    close_col      : Close price column name.
    time_col       : Timestamp column; falls back to df.index.
    zone_lo        : Lower Fibonacci bound (default 0.382).
    zone_hi        : Upper Fibonacci bound (default 0.786).
    require_struct : If True (default), zone entry alone is NOT a signal;
                     a 15M structure event in the bias direction is required.

    Returns
    -------
    SignalResult with all SignalEvent objects in chronological order.
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

    # Build a fast lookup: bar_index -> list[StructureEvent] in bias direction
    # (only events whose direction matches the 1H bias are relevant)
    struct_by_bar: dict[int, list[StructureEvent]] = {}
    for e in struct_res.events:
        if e.direction == bias:
            struct_by_bar.setdefault(e.index, []).append(e)

    events: list[SignalEvent] = []
    in_zone = False   # track whether we are currently inside the zone

    for i in range(n_bars):
        close = closes[i]

        in_full   = fib.is_in_zone(close, zone_lo, zone_hi)
        in_golden = fib.is_in_zone(close, 0.382, 0.618)
        in_deep   = fib.is_in_zone(close, 0.618, 0.786)
        nearest   = fib.nearest_level(close)

        def _make(kind: SignalKind,
                  struct: StructureEvent | None = None) -> SignalEvent:
            return SignalEvent(
                index=i,
                timestamp=timestamps[i],
                kind=kind,
                bias=bias,
                price=close,
                nearest_fib=nearest,
                in_golden=in_golden,
                in_deep=in_deep,
                struct_event=struct,
            )

        # ── Zone transitions ──────────────────────────────────────────────────
        if in_full and not in_zone:
            events.append(_make("zone_entry"))
            in_zone = True
        elif not in_full and in_zone:
            events.append(_make("zone_exit"))
            in_zone = False

        # ── Signal: structure confirmation inside the zone ────────────────────
        if in_zone:
            confirming = struct_by_bar.get(i, [])
            if confirming:
                # Use the first matching event (typically only one per bar)
                events.append(_make("signal", struct=confirming[0]))
            elif not require_struct:
                # Zone entry without structure confirmation
                events.append(_make("signal"))

    return SignalResult(
        events=events,
        fib=fib,
        bias=bias,
        zone_lo=zone_lo,
        zone_hi=zone_hi,
    )
