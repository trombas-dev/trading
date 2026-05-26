"""
fibonacci.py — Fibonacci retracement and extension calculator.

Given a swing high, a swing low, and the direction of the prevailing
move, this module computes:

  Retracement levels  23.6 %  38.2 %  50.0 %  61.8 %  78.6 %
  Extension levels   127.2 %  161.8 %   (TP1 / TP2 targets)

Direction convention
--------------------
  "bullish"  Market rallied from swing_low → swing_high.
             A retracement PULLS BACK toward the low.
             Measured as:  price = high - ratio * range
             Extensions project ABOVE the high (TP targets):
             price = high + (ratio - 1.0) * range

  "bearish"  Market fell from swing_high → swing_low.
             A retracement BOUNCES BACK toward the high.
             Measured as:  price = low + ratio * range
             Extensions project BELOW the low (TP targets):
             price = low - (ratio - 1.0) * range

Golden zone
-----------
  The 0.382–0.618 band is the primary entry zone; 0.5–0.786 is the
  "premium" / "deep" zone used for high-conviction reversal entries.
  is_in_zone() checks whether a price sits inside a configured band.

Variables for learning (step 3)
--------------------------------
  fib_entry_lo   : float, default 0.382
                   Lower bound of the entry zone.
                   Valid: 0.236, 0.382, 0.5
  fib_entry_hi   : float, default 0.618
                   Upper bound of the entry zone.
                   Valid: 0.618, 0.705, 0.786
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hermes_trading.strategy.swing import SwingPoint

# ── Public types ─────────────────────────────────────────────────────────────

Direction = Literal["bullish", "bearish"]
LevelKind = Literal["anchor", "retracement", "extension"]

# Standard ratios — do not modify; use fib_entry_lo/hi to narrow the zone.
RETRACEMENT_RATIOS: tuple[float, ...] = (0.236, 0.382, 0.500, 0.618, 0.786)
EXTENSION_RATIOS:   tuple[float, ...] = (1.272, 1.618)

_LABELS: dict[float, str] = {
    0.000: "0.0 %   (swing start)",
    0.236: "23.6 %",
    0.382: "38.2 %",
    0.500: "50.0 %",
    0.618: "61.8 %",
    0.786: "78.6 %",
    1.000: "100.0 % (swing end)",
    1.272: "TP1  127.2 %",
    1.618: "TP2  161.8 %",
}


@dataclass(frozen=True)
class FibLevel:
    """One Fibonacci price level."""

    ratio: float        # e.g. 0.382 or 1.272
    price: float        # actual price of this level
    label: str          # human-readable label
    kind:  LevelKind    # "anchor" | "retracement" | "extension"

    def __repr__(self) -> str:
        return (
            f"FibLevel({self.label:<22s}  "
            f"price={self.price:>12.4f}  "
            f"kind={self.kind})"
        )


@dataclass
class FibResult:
    """Complete set of Fibonacci levels for one swing move."""

    direction:  Direction
    swing_high: SwingPoint
    swing_low:  SwingPoint
    levels:     list[FibLevel]  # sorted high → low in price

    # ── Anchors ───────────────────────────────────────────────────────────────

    @property
    def high(self) -> float:
        return self.swing_high.price

    @property
    def low(self) -> float:
        return self.swing_low.price

    @property
    def range_usd(self) -> float:
        return self.high - self.low

    @property
    def range_pct(self) -> float:
        """Move size as a percentage of the low."""
        return self.range_usd / self.low * 100.0

    # ── Filtered views ────────────────────────────────────────────────────────

    @property
    def retracements(self) -> list[FibLevel]:
        """Anchor + retracement levels only (0 % … 100 %)."""
        return [l for l in self.levels if l.kind in ("anchor", "retracement")]

    @property
    def extensions(self) -> list[FibLevel]:
        """Extension levels sorted by ratio ascending: 127.2 % first, 161.8 % second.

        self.levels is sorted high -> low by price.  For a bearish move that
        order puts 127.2 % before 161.8 % (both below the low).  For a bullish
        move the high -> low order reverses them (161.8 % is above 127.2 %).
        Sorting by ratio makes tp1 / tp2 direction-independent.
        """
        return sorted(
            [l for l in self.levels if l.kind == "extension"],
            key=lambda lv: lv.ratio,
        )

    @property
    def tp1(self) -> FibLevel | None:
        """127.2 % extension (first TP target)."""
        exts = self.extensions
        return exts[0] if exts else None

    @property
    def tp2(self) -> FibLevel | None:
        """161.8 % extension (second TP target)."""
        exts = self.extensions
        return exts[1] if len(exts) > 1 else None

    # ── Zone helpers ──────────────────────────────────────────────────────────

    def zone(
        self,
        lo_ratio: float = 0.382,
        hi_ratio: float = 0.618,
    ) -> tuple[float, float]:
        """
        Price range of a Fibonacci band as (price_lo, price_hi).

        For bullish setups the band is BELOW the swing high.
        For bearish setups the band is ABOVE the swing low.
        """
        spread = self.range_usd
        if self.direction == "bullish":
            # Retracements pull down from the high
            price_hi = self.high - lo_ratio * spread
            price_lo = self.high - hi_ratio * spread
        else:
            # Retracements bounce up from the low
            price_lo = self.low + lo_ratio * spread
            price_hi = self.low + hi_ratio * spread
        return price_lo, price_hi

    def is_in_zone(
        self,
        price: float,
        lo_ratio: float = 0.382,
        hi_ratio: float = 0.618,
    ) -> bool:
        """True if `price` sits inside the (lo_ratio, hi_ratio) band."""
        lo, hi = self.zone(lo_ratio, hi_ratio)
        return lo <= price <= hi

    def nearest_level(self, price: float) -> FibLevel:
        """The FibLevel whose price is closest to `price`."""
        return min(self.levels, key=lambda lv: abs(lv.price - price))

    def level_at(self, ratio: float) -> FibLevel | None:
        """Return the FibLevel for an exact ratio (e.g. 0.618), or None."""
        for lv in self.levels:
            if abs(lv.ratio - ratio) < 1e-9:
                return lv
        return None

    def summary(self) -> str:
        tp1_str = f"${self.tp1.price:,.2f}" if self.tp1 else "—"
        tp2_str = f"${self.tp2.price:,.2f}" if self.tp2 else "—"
        return (
            f"FibResult  {self.direction}  "
            f"high=${self.high:,.2f}  low=${self.low:,.2f}  "
            f"range={self.range_pct:.2f}%  "
            f"TP1={tp1_str}  TP2={tp2_str}"
        )


# ── Core calculator ───────────────────────────────────────────────────────────

def calculate_fibs(
    swing_high: SwingPoint,
    swing_low:  SwingPoint,
    direction:  Direction,
    retracement_ratios: tuple[float, ...] = RETRACEMENT_RATIOS,
    extension_ratios:   tuple[float, ...] = EXTENSION_RATIOS,
) -> FibResult:
    """
    Calculate Fibonacci retracement and extension levels for a swing move.

    Parameters
    ----------
    swing_high         : SwingPoint at the top of the move.
    swing_low          : SwingPoint at the bottom of the move.
    direction          : "bullish" — rally from low to high (retraces down).
                         "bearish" — drop from high to low (retraces up).
    retracement_ratios : Ratios to compute between the two anchors.
    extension_ratios   : Ratios for projection beyond the move.

    Returns
    -------
    FibResult with all FibLevel objects sorted high → low by price.
    """
    if swing_high.price <= swing_low.price:
        raise ValueError(
            f"swing_high.price ({swing_high.price:.4f}) must be greater "
            f"than swing_low.price ({swing_low.price:.4f})"
        )

    high   = swing_high.price
    low    = swing_low.price
    spread = high - low

    levels: list[FibLevel] = []

    # ── Anchor levels (0 % and 100 %) ─────────────────────────────────────────
    levels.append(FibLevel(
        ratio=0.0,
        price=high if direction == "bullish" else low,
        label=_LABELS[0.0],
        kind="anchor",
    ))
    levels.append(FibLevel(
        ratio=1.0,
        price=low if direction == "bullish" else high,
        label=_LABELS[1.0],
        kind="anchor",
    ))

    # ── Retracement levels ────────────────────────────────────────────────────
    for r in retracement_ratios:
        if direction == "bullish":
            price = high - r * spread          # pulls back from high toward low
        else:
            price = low + r * spread           # bounces from low toward high

        levels.append(FibLevel(
            ratio=r,
            price=price,
            label=_LABELS.get(r, f"{r*100:.1f} %"),
            kind="retracement",
        ))

    # ── Extension levels ──────────────────────────────────────────────────────
    for r in extension_ratios:
        ext_fraction = r - 1.0                 # how far beyond the full move
        if direction == "bullish":
            price = high + ext_fraction * spread   # above the high
        else:
            price = low  - ext_fraction * spread   # below the low

        levels.append(FibLevel(
            ratio=r,
            price=price,
            label=_LABELS.get(r, f"EXT {r*100:.1f} %"),
            kind="extension",
        ))

    # Sort high → low
    levels.sort(key=lambda lv: lv.price, reverse=True)

    return FibResult(
        direction=direction,
        swing_high=swing_high,
        swing_low=swing_low,
        levels=levels,
    )


# ── Convenience: derive fibs from any structure event (BOS or CHoCH) ─────────

def fibs_from_structure_event(
    evt,            # StructureEvent  (BOS or CHoCH — logic is identical)
    swing_result,   # SwingResult from the SAME timeframe as evt
) -> "FibResult | None":
    """
    Build a FibResult from ONE structure event (BOS or CHoCH).

    The event's ``broken_swing`` is one anchor; the most recent swing on the
    opposite side (confirmed before the broken swing) is the other anchor.

    Works for both BOS and CHoCH because the broken_swing semantics are the
    same: bullish events break a swing HIGH, bearish events break a swing LOW.

    Used by the rolling backtester to compute per-window fib levels.
    """
    if evt.direction == "bullish":
        swing_high = evt.broken_swing
        candidates = [
            p for p in swing_result.lows
            if p.confirmed_at < swing_high.index
        ]
        if not candidates:
            return None
        swing_low = candidates[-1]
        direction: Direction = "bullish"
    else:
        swing_low = evt.broken_swing
        candidates = [
            p for p in swing_result.highs
            if p.confirmed_at < swing_low.index
        ]
        if not candidates:
            return None
        swing_high = candidates[-1]
        direction = "bearish"

    if swing_high.price <= swing_low.price:
        return None
    return calculate_fibs(swing_high, swing_low, direction)


# Backward-compat alias — callers that use the old name still work
fibs_from_choch_event = fibs_from_structure_event


def fibs_from_last_choch(
    struct_result,           # StructureResult
    swing_result,            # SwingResult
    as_of: int | None = None,
) -> FibResult | None:
    """
    Build a FibResult from the most recent CHoCH event.

    The CHoCH's broken_swing provides one anchor; the most recent swing on
    the opposite side (confirmed before the broken_swing) provides the other.

    Returns None if there is no CHoCH or opposite swing available.
    """
    choch = struct_result.last_choch(as_of=as_of)
    if choch is None:
        return None

    if choch.direction == "bullish":
        # Bullish CHoCH: broken_swing is a swing HIGH → high anchor.
        # Find the swing LOW confirmed before that high.
        swing_high = choch.broken_swing
        candidates = [
            p for p in swing_result.lows
            if p.confirmed_at < swing_high.index
        ]
        if not candidates:
            return None
        swing_low = candidates[-1]
        direction: Direction = "bullish"

    else:
        # Bearish CHoCH: broken_swing is a swing LOW → low anchor.
        # Find the swing HIGH confirmed before that low.
        swing_low = choch.broken_swing
        candidates = [
            p for p in swing_result.highs
            if p.confirmed_at < swing_low.index
        ]
        if not candidates:
            return None
        swing_high = candidates[-1]
        direction = "bearish"

    return calculate_fibs(swing_high, swing_low, direction)
