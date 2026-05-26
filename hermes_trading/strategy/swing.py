"""
swing.py — N-candle swing high / swing low detector.

Definition
----------
A swing HIGH at index i (with window N, lookback lb = (N-1)//2):
    high[i] > high[j]  for every j in [i-lb .. i+lb], j != i

A swing LOW at index i:
    low[i]  < low[j]   for every j in [i-lb .. i+lb], j != i

Ties disqualify the centre — only a strict unique peak/trough counts.

Confirmation lag
----------------
In a live system a swing at index i is only CONFIRMED once candle
i+lb has closed (we need lb candles to the right to verify).
SwingPoint.confirmed_at records that index so the backtester can
avoid lookahead bias.

Variables for learning
----------------------
  n_candles : int, odd, default 7
              Valid values: 3, 5, 7, 9, 11
              Larger N → fewer but stronger swing points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# ── Public types ────────────────────────────────────────────────────────────

Kind = Literal["high", "low"]


@dataclass(frozen=True)
class SwingPoint:
    """One confirmed swing high or swing low."""

    index: int                  # position in the source DataFrame
    confirmed_at: int           # index when the swing was confirmed (index + lb)
    timestamp: pd.Timestamp     # candle open time at `index`
    price: float                # high price (swing high) or low price (swing low)
    kind: Kind                  # "high" or "low"

    def __repr__(self) -> str:
        return (
            f"SwingPoint({self.kind.upper():4s}  "
            f"price={self.price:>12.4f}  "
            f"at={str(self.timestamp)[:16]}  "
            f"confirmed_at_bar={self.confirmed_at})"
        )


@dataclass
class SwingResult:
    """All swing points found in a single pass, with convenience accessors."""

    points: list[SwingPoint] = field(default_factory=list)
    n_candles: int = 7

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def highs(self) -> list[SwingPoint]:
        return [p for p in self.points if p.kind == "high"]

    @property
    def lows(self) -> list[SwingPoint]:
        return [p for p in self.points if p.kind == "low"]

    def confirmed_by(self, bar_index: int) -> list[SwingPoint]:
        """Points that were confirmed on or before `bar_index`."""
        return [p for p in self.points if p.confirmed_at <= bar_index]

    def last_high(self, as_of: int | None = None) -> SwingPoint | None:
        pool = self.confirmed_by(as_of) if as_of is not None else self.points
        highs = [p for p in pool if p.kind == "high"]
        return highs[-1] if highs else None

    def last_low(self, as_of: int | None = None) -> SwingPoint | None:
        pool = self.confirmed_by(as_of) if as_of is not None else self.points
        lows = [p for p in pool if p.kind == "low"]
        return lows[-1] if lows else None

    def last_n_highs(self, n: int, as_of: int | None = None) -> list[SwingPoint]:
        pool = self.confirmed_by(as_of) if as_of is not None else self.points
        return [p for p in pool if p.kind == "high"][-n:]

    def last_n_lows(self, n: int, as_of: int | None = None) -> list[SwingPoint]:
        pool = self.confirmed_by(as_of) if as_of is not None else self.points
        return [p for p in pool if p.kind == "low"][-n:]

    def summary(self) -> str:
        return (
            f"SwingResult  n={self.n_candles}  "
            f"total={len(self.points)}  "
            f"highs={len(self.highs)}  "
            f"lows={len(self.lows)}"
        )


# ── Core detector ────────────────────────────────────────────────────────────

def find_swings(
    df: pd.DataFrame,
    n: int = 7,
    high_col: str = "high",
    low_col: str = "low",
    time_col: str | None = None,
) -> SwingResult:
    """
    Detect all swing highs and lows in an OHLCV DataFrame.

    Parameters
    ----------
    df        : DataFrame with at least `high_col` and `low_col` columns.
                Index should be DatetimeIndex (or set `time_col`).
    n         : Window size — must be odd and >= 3.
                lb = (n-1)//2 candles required on each side.
    high_col  : Column name for candle highs.
    low_col   : Column name for candle lows.
    time_col  : If set, use this column for timestamps instead of the index.

    Returns
    -------
    SwingResult with all SwingPoint objects, sorted by index.
    """
    _validate_n(n)
    lb = (n - 1) // 2

    if len(df) < n:
        return SwingResult(points=[], n_candles=n)

    highs = df[high_col].to_numpy(dtype=float)
    lows  = df[low_col].to_numpy(dtype=float)

    if time_col:
        timestamps = pd.to_datetime(df[time_col])
    else:
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) \
                     else pd.RangeIndex(len(df))

    points: list[SwingPoint] = []

    for i in range(lb, len(df) - lb):
        window_h = highs[i - lb : i + lb + 1]
        window_l = lows[i - lb  : i + lb + 1]

        # Swing HIGH: centre is the strict unique maximum
        if highs[i] == window_h.max() and int((window_h == highs[i]).sum()) == 1:
            points.append(SwingPoint(
                index=i,
                confirmed_at=i + lb,
                timestamp=timestamps[i],
                price=float(highs[i]),
                kind="high",
            ))

        # Swing LOW: centre is the strict unique minimum
        if lows[i] == window_l.min() and int((window_l == lows[i]).sum()) == 1:
            points.append(SwingPoint(
                index=i,
                confirmed_at=i + lb,
                timestamp=timestamps[i],
                price=float(lows[i]),
                kind="low",
            ))

    points.sort(key=lambda p: p.index)
    return SwingResult(points=points, n_candles=n)


# ── Helpers ──────────────────────────────────────────────────────────────────

def valid_n_values() -> list[int]:
    """Allowed values for the n_candles learning variable."""
    return [3, 5, 7, 9, 11]


def _validate_n(n: int) -> None:
    if n not in valid_n_values():
        raise ValueError(
            f"n must be one of {valid_n_values()}, got {n}. "
            f"n must be odd and between 3–11."
        )
