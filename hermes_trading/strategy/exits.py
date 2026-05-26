"""
exits.py — Virtual SL + TP management (close-based).

Pipeline position
-----------------
  1H  -> fib / bias
  15M -> signals
  5M  -> entries
  5M  -> exits   <- this module

Close-based SL rule
-------------------
Stop loss is only evaluated on candle CLOSE — intra-bar wicks that pierce
the SL level but recover before the close do NOT trigger a stop.

Two-phase trade lifecycle
--------------------------
Phase 1  Full position (100 %)
  Bullish                           Bearish
  close <= SL   -> STOP_LOSS       close >= SL   -> STOP_LOSS
  close >= TP1  -> TP1_HIT         close <= TP1  -> TP1_HIT
    50 % of position closed                  50 % closed
    SL moved to entry_price (BE)             SL moved to entry_price (BE)
    -> enter Phase 2                         -> enter Phase 2

Phase 2  Half position (50 %) with SL at breakeven
  Bullish                           Bearish
  close <= entry -> BREAKEVEN      close >= entry -> BREAKEVEN  (0 R on half)
  close >= TP2   -> TP2_HIT        close <= TP2   -> TP2_HIT
  i > max_bars   -> TIMEOUT        i > max_bars   -> TIMEOUT

PnL in R
--------
  STOP_LOSS              :   -1.00 R
  TP1 + BREAKEVEN        :   +0.5 * rr1    (rr1 = (tp1-entry)/risk)
  TP1 + TP2              :   +0.5 * rr1  +  0.5 * rr2
  TIMEOUT (phase 1)      :   (close - entry) / risk
  TIMEOUT (phase 2)      :   0.5*rr1  +  0.5*(close-entry)/risk

Variables for learning
----------------------
  max_bars_in_trade : int  default 100  (100 * 5 min = ~8 hours)
                      Valid: 50, 100, 150, 200
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from hermes_trading.strategy.entry import EntryResult, EntrySignal

# ── Public types ─────────────────────────────────────────────────────────────

ExitKind  = Literal["stop_loss", "tp1", "breakeven", "tp2", "timeout"]
Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class ExitEvent:
    """One partial or full close event."""

    index:     int               # 5M bar index
    timestamp: pd.Timestamp
    kind:      ExitKind
    price:     float             # price at which the exit is filled
    fraction:  float             # share of position closed (0.5 or 1.0)
    pnl_r:     float             # PnL in R for this fraction

    def __repr__(self) -> str:
        return (
            f"ExitEvent({self.kind:<11s}  "
            f"price={self.price:>12.4f}  "
            f"frac={self.fraction:.1f}  "
            f"pnl={self.pnl_r:+.3f}R  "
            f"at={str(self.timestamp)[:16]})"
        )


@dataclass
class TradeResult:
    """Complete lifecycle of one trade from entry to final close."""

    entry: EntrySignal
    exits: list[ExitEvent] = field(default_factory=list)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def total_pnl_r(self) -> float:
        return sum(e.pnl_r for e in self.exits)

    @property
    def closed_fraction(self) -> float:
        return sum(e.fraction for e in self.exits)

    @property
    def is_closed(self) -> bool:
        return self.closed_fraction >= 1.0 - 1e-9

    @property
    def is_win(self) -> bool:
        return self.total_pnl_r > 1e-9

    @property
    def status(self) -> str:
        """'open' | 'win' | 'loss' | 'breakeven' | 'partial'"""
        if not self.exits:
            return "open"
        if not self.is_closed:
            return "partial"
        r = self.total_pnl_r
        if r > 0.05:
            return "win"
        if r < -0.05:
            return "loss"
        return "breakeven"

    @property
    def duration_bars(self) -> int:
        """Number of 5M bars from entry to last exit."""
        if not self.exits:
            return 0
        return self.exits[-1].index - self.entry.index

    @property
    def final_exit(self) -> ExitEvent | None:
        return self.exits[-1] if self.exits else None

    def __repr__(self) -> str:
        return (
            f"TradeResult({self.status:<9s}  "
            f"R={self.total_pnl_r:+.3f}  "
            f"entry=${self.entry.entry_price:,.2f}  "
            f"at={str(self.entry.timestamp)[:16]}  "
            f"bars={self.duration_bars})"
        )


@dataclass
class ExitResult:
    """Aggregate of all simulated trades."""

    trades:       list[TradeResult] = field(default_factory=list)
    max_bars:     int               = 100

    # ── Filters ───────────────────────────────────────────────────────────────

    @property
    def closed(self) -> list[TradeResult]:
        return [t for t in self.trades if t.is_closed]

    @property
    def wins(self) -> list[TradeResult]:
        return [t for t in self.closed if t.is_win]

    @property
    def losses(self) -> list[TradeResult]:
        return [t for t in self.closed if t.status == "loss"]

    @property
    def breakevens(self) -> list[TradeResult]:
        return [t for t in self.closed if t.status == "breakeven"]

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        c = self.closed
        return len(self.wins) / len(c) * 100 if c else 0.0

    @property
    def total_pnl_r(self) -> float:
        return sum(t.total_pnl_r for t in self.closed)

    @property
    def avg_pnl_r(self) -> float:
        c = self.closed
        return self.total_pnl_r / len(c) if c else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win  = sum(t.total_pnl_r for t in self.wins)
        gross_loss = abs(sum(t.total_pnl_r for t in self.losses))
        if gross_loss < 1e-9:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def max_win_r(self) -> float:
        return max((t.total_pnl_r for t in self.wins), default=0.0)

    @property
    def max_loss_r(self) -> float:
        return min((t.total_pnl_r for t in self.losses), default=0.0)

    @property
    def equity_curve(self) -> list[float]:
        """Cumulative R after each closed trade (for charting)."""
        curve, total = [], 0.0
        for t in self.closed:
            total += t.total_pnl_r
            curve.append(total)
        return curve

    def summary(self) -> str:
        return (
            f"ExitResult  "
            f"trades={len(self.trades)}  "
            f"closed={len(self.closed)}  "
            f"wins={len(self.wins)}  "
            f"losses={len(self.losses)}  "
            f"BE={len(self.breakevens)}  "
            f"WR={self.win_rate:.1f}%  "
            f"totalR={self.total_pnl_r:+.2f}  "
            f"avgR={self.avg_pnl_r:+.2f}  "
            f"PF={self.profit_factor:.2f}"
        )


# ── Core simulator ────────────────────────────────────────────────────────────

def simulate_exits(
    df: pd.DataFrame,
    entries: EntryResult,
    close_col:   str   = "close",
    high_col:    str   = "high",
    low_col:     str   = "low",
    time_col:    str | None = None,
    max_bars:    int   = 100,
    half_spread: float = 0.0,
) -> ExitResult:
    """
    Simulate trade exits for every entry in `entries`.

    Parameters
    ----------
    df           : 5M OHLCV DataFrame (same one used for entries).
    entries      : EntryResult from find_entries().
    close_col    : Close price column (SL evaluated on close).
    high_col     : High price column.
    low_col      : Low price column.
    time_col     : Timestamp column; falls back to df.index.
    max_bars     : Maximum 5M bars a trade can stay open before timeout.
    half_spread  : Half the bid-ask spread in price units (e.g. 0.00015 for
                   EURUSD at 3 pip spread).  Applied as:
                     Long  entry : effective_ep = ep + half_spread  (buy at ask)
                     Short entry : effective_ep = ep - half_spread  (sell at bid)
                     All exits   : effective_exit worsened by half_spread
                   Produces a realistic round-trip cost deduction per trade.
                   Set to 0.0 (default) for no spread simulation.

    Returns
    -------
    ExitResult with one TradeResult per entry, in chronological order.
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

    trades: list[TradeResult] = []

    for entry in entries.entries:
        direction = entry.direction
        ep        = entry.entry_price
        sl_price  = entry.stop_loss
        tp1_price = entry.tp1
        tp2_price = entry.tp2

        # ── Spread adjustment ─────────────────────────────────────────────────
        # Longs: you buy at ask (ep + half_spread); shorts: sell at bid (ep - half_spread).
        # All exits are also worsened by half_spread (sell at bid / buy at ask).
        if half_spread > 0.0:
            ep_eff   = ep + half_spread if direction == "bullish" else ep - half_spread
            risk_eff = max(abs(ep_eff - sl_price), 1e-8)
        else:
            ep_eff   = ep
            risk_eff = entry.risk

        exits:   list[ExitEvent] = []
        phase    = 1              # 1 = full, 2 = half (SL at BE)
        sl_live  = sl_price       # tracks BE move
        tp1_hit  = False

        def _pnl(exit_px: float, frac: float) -> float:
            # Worsen the exit price by half_spread (pay spread on the way out too)
            if direction == "bullish":
                eff = exit_px - half_spread
                return frac * (eff - ep_eff) / risk_eff
            else:
                eff = exit_px + half_spread
                return frac * (ep_eff - eff) / risk_eff

        start = entry.index + 1
        end   = min(entry.index + max_bars + 1, n_bars)

        closed = False

        for i in range(start, end):
            close = closes[i]
            ts    = timestamps[i]

            # ── Phase 1: full position ────────────────────────────────────────
            if phase == 1:
                # Close-based SL check
                sl_triggered = (close <= sl_live) if direction == "bullish" \
                               else (close >= sl_live)
                tp1_triggered = (close >= tp1_price) if direction == "bullish" \
                                else (close <= tp1_price)

                if sl_triggered:
                    exits.append(ExitEvent(i, ts, "stop_loss",
                                           sl_live, 1.0, _pnl(sl_live, 1.0)))
                    closed = True
                    break

                if tp1_triggered:
                    exits.append(ExitEvent(i, ts, "tp1",
                                           tp1_price, 0.5, _pnl(tp1_price, 0.5)))
                    sl_live  = ep        # move SL to breakeven
                    tp1_hit  = True
                    phase    = 2

            # ── Phase 2: half position, SL at breakeven ───────────────────────
            if phase == 2:
                be_triggered  = (close <= sl_live) if direction == "bullish" \
                                else (close >= sl_live)
                tp2_triggered = (close >= tp2_price) if direction == "bullish" \
                                else (close <= tp2_price)

                if be_triggered:
                    exits.append(ExitEvent(i, ts, "breakeven",
                                           sl_live, 0.5, _pnl(sl_live, 0.5)))
                    closed = True
                    break

                if tp2_triggered:
                    exits.append(ExitEvent(i, ts, "tp2",
                                           tp2_price, 0.5, _pnl(tp2_price, 0.5)))
                    closed = True
                    break

        # ── Timeout: close whatever fraction is still open ────────────────────
        if not closed:
            i     = min(end - 1, n_bars - 1)
            close = closes[i]
            ts    = timestamps[i]
            frac  = 0.5 if tp1_hit else 1.0
            exits.append(ExitEvent(i, ts, "timeout", close, frac, _pnl(close, frac)))

        trades.append(TradeResult(entry=entry, exits=exits))

    return ExitResult(trades=trades, max_bars=max_bars)
