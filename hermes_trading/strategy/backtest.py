"""
backtest.py — Rolling walk-forward backtest engine + metrics + optimiser.

Pipeline per CHoCH window
--------------------------
For every 1H CHoCH event the engine opens a trading window that lasts
until the NEXT 1H CHoCH.  Inside each window it runs:
  1H fib + bias  ->  15M signals  ->  5M entries  ->  5M exits

This means fib levels are recomputed automatically whenever the market
structure changes — no static zones, no look-ahead.

Walk-forward splits
-------------------
  in_sample   : IS_START  …  IS_END    (parameter fitting)
  validation  : IS_END+1  …  VAL_END   (sanity check, NOT used for fitting)
  out_of_sample: VAL_END+1 … OOS_END   (final performance)

Optimiser
---------
Grid-searches the parameter space on IN-SAMPLE data only.
Best parameter set is selected by Sharpe ratio (min 10 trades required).
The selected set is then evaluated on VALIDATION and OOS without change.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from hermes_trading.strategy.swing import find_swings
from hermes_trading.strategy.structure import find_structure
from hermes_trading.strategy.fibonacci import (
    fibs_from_structure_event, fibs_from_choch_event,
    calculate_fibs, FibResult,
)
from hermes_trading.strategy.signal import find_signals
from hermes_trading.strategy.entry import find_entries
from hermes_trading.strategy.exits import simulate_exits, ExitResult, TradeResult
from hermes_trading.strategy.regime import compute_regime_ok


# ── Scenario definitions ──────────────────────────────────────────────────────

# Maps scenario_id -> (tf_bias, tf_signal, tf_entry)
# Scenario 1: 1H CHoCH drives windows; 15M signals; 5M entries  (default)
# Scenario 2: 4H CHoCH drives windows; 1H signals; 15M entries  (medium swing)
# Scenario 3: 1D CHoCH drives windows; 4H signals; 1H entries   (daily swing)
SCENARIO_TFS: dict[int, tuple[str, str, str]] = {
    1: ("1H",  "15M", "5M"),
    2: ("4H",  "1H",  "15M"),
    3: ("1D",  "4H",  "1H"),
}

SCENARIO_NAMES: dict[int, str] = {
    1: "Scenario 1  1H/15M/5M  (default)",
    2: "Scenario 2  4H/1H/15M  (medium swing)",
    3: "Scenario 3  1D/4H/1H   (daily swing)",
}


# ── Parameter set ─────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    n_bias:            int   = 7
    n_signal:          int   = 5
    n_entry:           int   = 3
    zone_lo:           float = 0.382
    zone_hi:           float = 0.786
    timeout_bars:      int   = 4    # entry-TF bars window stays open after signal
    max_bars_in_trade: int   = 100  # entry-TF bars max trade duration
    scenario:          int   = 1    # 1=1H/15M/5M | 2=4H/1H/15M | 3=1D/4H/1H
    use_bos_windows:   bool  = False  # also open windows on BOS (not only CHoCH)
    # How fib levels are drawn when a BOS opens a window (use_bos_windows=True):
    #   "last_leg"   — fib from last pullback low → BOS level  (default, same as CHoCH logic)
    #   "full_trend" — fib from CHoCH origin      → BOS level  (wider zone, trend-anchored)
    #   "delayed"    — wait for next swing high after BOS, then fib last_low → new_high
    bos_fib_mode:      str   = "last_leg"
    # Spread simulation: half the round-trip bid-ask spread in price units.
    # E.g. EURUSD 2-pip spread → half_spread = 0.0001
    #      USDJPY 3-pip spread → half_spread = 0.015
    #      GBPJPY 4-pip spread → half_spread = 0.02
    # Set to 0.0 to disable (default, matches original backtest results).
    half_spread:       float = 0.0

    # ── Regime filter ─────────────────────────────────────────────────────────
    # Skips bias-TF windows that open during choppy / ranging market regimes.
    # The indicator is computed on `regime_tf` (default "1D") and evaluated at
    # each window's ts_start using the last known value (no lookahead).
    #
    # regime_filter options
    #   "none"       Disabled — all windows allowed (default; reproduces original results)
    #   "adx"        Skip when ADX(regime_adx_period) < regime_adx_min on regime_tf
    #                Wilder's thresholds: <20 = choppy, 20-25 = weak, >25 = trending
    #   "atr_ratio"  Skip when ATR(atr_short)/ATR(atr_long) < atr_ratio_min
    #                Catches low-volatility compression zones
    regime_filter:      str   = "none"   # "none" | "adx" | "atr_ratio"
    regime_tf:          str   = "1D"     # TF to compute indicator on (must be in dfs)
    regime_adx_period:  int   = 14       # ADX smoothing period (Wilder's standard)
    regime_adx_min:     float = 20.0     # minimum ADX to allow trading
    regime_atr_short:   int   = 10       # short ATR period  (atr_ratio filter)
    regime_atr_long:    int   = 50       # long  ATR period  (atr_ratio filter)
    regime_atr_ratio:   float = 0.80     # min ATR_short/ATR_long ratio

    def tfs(self) -> tuple[str, str, str]:
        """Return (tf_bias, tf_signal, tf_entry) for this scenario."""
        return SCENARIO_TFS.get(self.scenario, SCENARIO_TFS[1])

    def label(self) -> str:
        tf_bias, tf_sig, tf_ent = self.tfs()
        if self.use_bos_windows:
            short = {"last_leg": "ll", "full_trend": "ft", "delayed": "dl"}
            bos_tag = f"_bos_{short.get(self.bos_fib_mode, self.bos_fib_mode)}"
        else:
            bos_tag = ""
        if self.regime_filter == "none":
            regime_tag = ""
        elif self.regime_filter == "adx":
            regime_tag = f"_radx{self.regime_adx_min:.0f}"
        elif self.regime_filter == "atr_ratio":
            regime_tag = f"_ratr{int(self.regime_atr_ratio * 100)}"
        else:
            regime_tag = f"_r{self.regime_filter[:4]}"
        return (
            f"s{self.scenario}[{tf_bias}/{tf_sig}/{tf_ent}]{bos_tag}{regime_tag}_"
            f"nb{self.n_bias}_ns{self.n_signal}_ne{self.n_entry}"
            f"_z{self.zone_lo:.2f}-{self.zone_hi:.2f}"
            f"_t{self.timeout_bars}_m{self.max_bars_in_trade}"
        )


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(ex: ExitResult, trading_days: float = 252.0) -> dict[str, Any]:
    """
    Compute standard backtest metrics from an ExitResult.

    All return/risk metrics are expressed in R (1R = 1x the per-trade risk).
    CAGR and Sharpe assume each trade risks 1 % of account equity.
    """
    closed = ex.closed
    n      = len(closed)
    if n == 0:
        return {"n_trades": 0}

    pnls  = [t.total_pnl_r for t in closed]
    wins  = [p for p in pnls if p > 0.0]
    losses= [p for p in pnls if p < 0.0]

    total_r    = sum(pnls)
    avg_r      = total_r / n
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))

    # Equity curve in R
    eq = list(itertools.accumulate(pnls))
    peak = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualised, trade-level)
    std_r = float(np.std(pnls, ddof=1)) if n > 1 else 0.0
    # Estimate trades per year from duration
    if closed and closed[0].entry.timestamp != closed[-1].entry.timestamp:
        span_days = (
            closed[-1].entry.timestamp - closed[0].entry.timestamp
        ).total_seconds() / 86400
        trades_per_year = n / span_days * trading_days if span_days > 0 else 0
    else:
        trades_per_year = trading_days
    sharpe = (avg_r / std_r * (trades_per_year ** 0.5)) if std_r > 1e-9 else 0.0

    # CAGR (1 % risk per trade, $10k start)
    account_r = total_r * 0.01  # total return as fraction of account
    if closed:
        span_days = max(1.0, (
            closed[-1].entry.timestamp - closed[0].entry.timestamp
        ).total_seconds() / 86400)
        cagr = (1 + account_r) ** (365 / span_days) - 1
    else:
        cagr = 0.0

    return {
        "n_trades":      n,
        "n_wins":        len(wins),
        "n_losses":      len(losses),
        "win_rate":      len(wins) / n * 100,
        "total_r":       round(total_r, 4),
        "avg_r":         round(avg_r, 4),
        "std_r":         round(std_r, 4),
        "sharpe":        round(sharpe, 3),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 1e-9 else float("inf"),
        "max_dd_r":      round(max_dd, 4),
        "max_win_r":     round(max(pnls), 4),
        "max_loss_r":    round(min(pnls), 4),
        "cagr_pct":      round(cagr * 100, 2),
        "equity_curve":  eq,
    }


# ── Regime filter application ────────────────────────────────────────────────

def _apply_regime_filter(
    windows:    list[dict],
    regime_ok:  pd.Series,
) -> list[dict]:
    """
    Remove windows where the regime indicator says 'choppy' at ts_start.

    Uses pd.Series.asof(ts) to find the last known regime state at or before
    each window's start time — no lookahead bias.

    Parameters
    ----------
    windows    : List of window dicts produced by _build_windows().
    regime_ok  : Boolean Series from compute_regime_ok() on the regime TF.
                 True = trending (allow), False = choppy (skip).

    Returns
    -------
    Filtered list with choppy-regime windows removed.
    """
    if regime_ok.empty or bool(regime_ok.all()):
        return windows

    # Ensure the regime index has timezone info
    idx = regime_ok.index
    if idx.tzinfo is None:
        regime_ok = regime_ok.copy()
        regime_ok.index = idx.tz_localize("UTC")

    filtered: list[dict] = []
    for win in windows:
        ts = win["ts_start"]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        try:
            val = regime_ok.asof(ts)
            # asof() returns NaN when ts is before any index value → fail open
            ok = bool(val) if pd.notna(val) else True
        except Exception:
            ok = True  # fail open on any unexpected error
        if ok:
            filtered.append(win)

    return filtered


# ── Window builder ───────────────────────────────────────────────────────────

def _build_windows(
    window_events: list,
    sw_bias,
    df_bias: pd.DataFrame,
    params: "BacktestParams",
    ts_end_of_data: pd.Timestamp,
) -> list[dict]:
    """
    Convert a list of bias-TF structure events into trade windows.

    Each window dict has:
      ts_start : pd.Timestamp   when the window becomes active
      ts_end   : pd.Timestamp   when the window expires (next event or end of data)
      fib      : FibResult      fib levels for this window
      bias     : str            "bullish" | "bearish"

    How fib levels are built per event kind and bos_fib_mode
    ---------------------------------------------------------
    CHoCH (always):
      fibs_from_structure_event — same as original: last_low → CHoCH_high (bullish)

    BOS + bos_fib_mode="last_leg"  (default):
      Same call as CHoCH: last_low → BOS_high.
      Measures the final impulse leg before the BOS.

    BOS + bos_fib_mode="full_trend":
      Uses the origin of the CURRENT bullish trend (the CHoCH's swing_low) as
      the fib low, and the BOS level as the fib high.
      Zone covers the entire move from the structural reversal to the new high.

    BOS + bos_fib_mode="delayed":
      Does NOT activate at BOS time.  Scans forward for the first new swing
      high (bullish) / low (bearish) confirmed AFTER the BOS that exceeds the
      BOS level.  Fib is drawn from the last pullback low to that new high.
      Window start is shifted to when that new swing is confirmed on the bias TF.
      If no new swing forms before the next event, the window is skipped.
    """
    from hermes_trading.strategy.swing import SwingPoint  # avoid circular

    windows: list[dict] = []

    # Running CHoCH anchors — updated each time a CHoCH fires
    choch_low:  "SwingPoint | None" = None   # bullish trend origin
    choch_high: "SwingPoint | None" = None   # bearish trend origin

    n = len(window_events)

    for idx, evt in enumerate(window_events):
        ts_end = (
            window_events[idx + 1].timestamp
            if idx < n - 1
            else ts_end_of_data
        )

        # ── Determine fib and ts_start based on event kind + mode ────────────
        fib      = None
        ts_start = evt.timestamp

        if evt.kind == "choch":
            # Always use the standard last-leg fib for CHoCH
            fib = fibs_from_structure_event(evt, sw_bias)
            # Remember this CHoCH's anchors for subsequent BOS events
            if fib is not None:
                if evt.direction == "bullish":
                    choch_low  = fib.swing_low
                else:
                    choch_high = fib.swing_high

        elif evt.kind == "bos":
            mode = params.bos_fib_mode

            if mode == "last_leg":
                # Same as CHoCH: last pullback → BOS swing high/low
                fib = fibs_from_structure_event(evt, sw_bias)

            elif mode == "full_trend":
                # Anchor to the CHoCH origin that started this trend leg
                if evt.direction == "bullish" and choch_low is not None:
                    sh = evt.broken_swing          # BOS swing high
                    sl = choch_low                  # CHoCH origin low
                    if sh.price > sl.price:
                        fib = calculate_fibs(sh, sl, "bullish")
                elif evt.direction == "bearish" and choch_high is not None:
                    sl = evt.broken_swing          # BOS swing low
                    sh = choch_high                 # CHoCH origin high
                    if sh.price > sl.price:
                        fib = calculate_fibs(sh, sl, "bearish")
                # Fallback if no CHoCH anchor yet
                if fib is None:
                    fib = fibs_from_structure_event(evt, sw_bias)

            elif mode == "delayed":
                # Wait for next swing high (bullish) / low (bearish) after the BOS
                bos_bar   = evt.index
                bos_price = evt.broken_swing.price

                if evt.direction == "bullish":
                    # First new swing high > BOS level confirmed after the BOS bar
                    candidates = [
                        h for h in sw_bias.highs
                        if h.confirmed_at > bos_bar and h.price > bos_price
                    ]
                    if not candidates:
                        continue                         # no new high → skip
                    h3 = candidates[0]
                    # Last swing low confirmed before H3's index (not confirmed_at)
                    lows_before = [
                        l for l in sw_bias.lows
                        if l.confirmed_at < h3.index
                    ]
                    if not lows_before:
                        continue
                    l_last = lows_before[-1]
                    if h3.price <= l_last.price:
                        continue
                    fib      = calculate_fibs(h3, l_last, "bullish")
                    # Window starts when H3 is confirmed on the bias TF
                    conf_idx = h3.confirmed_at
                    ts_start = (
                        df_bias.index[conf_idx]
                        if conf_idx < len(df_bias)
                        else ts_end_of_data
                    )

                else:  # bearish
                    candidates = [
                        l for l in sw_bias.lows
                        if l.confirmed_at > bos_bar and l.price < bos_price
                    ]
                    if not candidates:
                        continue
                    l3 = candidates[0]
                    highs_before = [
                        h for h in sw_bias.highs
                        if h.confirmed_at < l3.index
                    ]
                    if not highs_before:
                        continue
                    h_last = highs_before[-1]
                    if h_last.price <= l3.price:
                        continue
                    fib      = calculate_fibs(h_last, l3, "bearish")
                    conf_idx = l3.confirmed_at
                    ts_start = (
                        df_bias.index[conf_idx]
                        if conf_idx < len(df_bias)
                        else ts_end_of_data
                    )

        # ── Validate and append ──────────────────────────────────────────────
        if fib is None or fib.range_pct < 0.05:
            continue
        if ts_start >= ts_end:
            continue   # delayed start landed past window boundary

        windows.append({
            "ts_start": ts_start,
            "ts_end":   ts_end,
            "fib":      fib,
            "bias":     evt.direction,
        })

    return windows


# ── Core engine ───────────────────────────────────────────────────────────────

def run_single_backtest(
    df_bias_or_dfs,
    df_signal: pd.DataFrame | None = None,
    df_entry:  pd.DataFrame | None = None,
    params:    BacktestParams | None = None,
) -> ExitResult:
    """
    Run the full pipeline on a dataset using rolling CHoCH windows.

    Calling conventions
    -------------------
    New (recommended):
        run_single_backtest(dfs, params)
        where dfs is a dict mapping TF strings to DataFrames,
        e.g. {"1H": df_1h, "15M": df_15m, "5M": df_5m, "4H": df_4h, "1D": df_1d}

    Legacy (still supported):
        run_single_backtest(df_1h, df_15m, df_5m, params)

    The active timeframe triplet is read from params.scenario:
        1  ->  df_bias="1H", df_signal="15M", df_entry="5M"   (default)
        2  ->  df_bias="4H", df_signal="1H",  df_entry="15M"
        3  ->  df_bias="1D", df_signal="4H",  df_entry="1H"
    """
    if params is None:
        params = BacktestParams()

    # ── Resolve DataFrames from dict or positional args ───────────────────────
    if isinstance(df_bias_or_dfs, dict):
        dfs = df_bias_or_dfs
    else:
        # Legacy positional call: build dict with scenario-1 names
        dfs = {"1H": df_bias_or_dfs, "15M": df_signal, "5M": df_entry}

    tf_bias, tf_sig, tf_ent = params.tfs()
    df_bias   = dfs[tf_bias]
    df_signal = dfs[tf_sig]
    df_entry  = dfs[tf_ent]

    # ── Bias TF: detect swings + structure once ───────────────────────────────
    sw_bias = find_swings(df_bias, n=params.n_bias)
    st_bias = find_structure(df_bias, sw_bias)

    all_trades: list[TradeResult] = []

    ts_end_of_data = (
        df_bias.index[-1].tz_localize("UTC")
        if df_bias.index[-1].tzinfo is None
        else df_bias.index[-1]
    )

    # Select raw event list: CHoCH-only or CHoCH+BOS
    raw_events = st_bias.events if params.use_bos_windows else st_bias.choch

    # Build windows (applies fib mode logic for BOS events)
    windows = _build_windows(raw_events, sw_bias, df_bias, params, ts_end_of_data)

    # ── Regime filter: skip windows that open during choppy/ranging periods ────
    if params.regime_filter != "none" and params.regime_tf in dfs:
        regime_ok = compute_regime_ok(
            dfs[params.regime_tf],
            params.regime_filter,
            adx_period    = params.regime_adx_period,
            adx_min       = params.regime_adx_min,
            atr_short     = params.regime_atr_short,
            atr_long      = params.regime_atr_long,
            atr_ratio_min = params.regime_atr_ratio,
        )
        windows = _apply_regime_filter(windows, regime_ok)

    for win in windows:
        fib      = win["fib"]
        bias     = win["bias"]
        ts_start = win["ts_start"]
        ts_end   = win["ts_end"]

        # ── Slice signal + entry TFs to window ────────────────────────────────
        ws = df_signal[(df_signal.index >= ts_start) & (df_signal.index < ts_end)].copy()
        we = df_entry[ (df_entry.index  >= ts_start) & (df_entry.index  < ts_end)].copy()

        min_bars_sig = params.n_signal * 4
        min_bars_ent = params.n_entry  * 4
        if len(ws) < min_bars_sig or len(we) < min_bars_ent:
            continue

        # ── Signal TF pipeline ────────────────────────────────────────────────
        try:
            sw_sig = find_swings(ws, n=params.n_signal)
            st_sig = find_structure(ws, sw_sig)
            sig    = find_signals(
                ws, fib=fib, bias=bias, struct_res=st_sig,
                zone_lo=params.zone_lo, zone_hi=params.zone_hi,
            )
        except Exception:
            continue

        if not sig.signals:
            continue

        # ── Entry TF pipeline ─────────────────────────────────────────────────
        try:
            sw_ent = find_swings(we, n=params.n_entry)
            st_ent = find_structure(we, sw_ent)
            ent    = find_entries(
                we, sig_res=sig, struct_res=st_ent, fib=fib,
                timeout_bars=params.timeout_bars,
            )
            ex_w   = simulate_exits(we, ent, max_bars=params.max_bars_in_trade,
                                    half_spread=params.half_spread)
        except Exception:
            continue

        all_trades.extend(ex_w.trades)

    return ExitResult(trades=all_trades, max_bars=params.max_bars_in_trade)


# ── Walk-forward ──────────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """Results across in-sample / validation / OOS splits."""

    params:     BacktestParams
    is_result:  ExitResult
    val_result: ExitResult
    oos_result: ExitResult
    is_metrics:  dict
    val_metrics: dict
    oos_metrics: dict

    def print_summary(self) -> None:
        """Print a compact comparison table."""
        from rich.table import Table
        from rich import box
        from rich.console import Console

        t = Table(title="Walk-Forward Summary", box=box.SIMPLE_HEAD,
                  show_edge=False, title_style="bold")
        cols = ["Period", "Trades", "WR%", "Total R", "Avg R",
                "Sharpe", "MaxDD R", "PF", "CAGR%"]
        for c in cols:
            t.add_column(c, justify="right")

        for label, m in [("In-sample",   self.is_metrics),
                         ("Validation",  self.val_metrics),
                         ("Out-of-sample", self.oos_metrics)]:
            if not m.get("n_trades"):
                t.add_row(label, "0", *["-"] * (len(cols) - 2))
                continue
            pf = m["profit_factor"]
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            sharpe_color = "green" if m["sharpe"] > 1.0 else "yellow"
            t.add_row(
                label,
                str(m["n_trades"]),
                f"{m['win_rate']:.1f}",
                f"{m['total_r']:+.2f}",
                f"{m['avg_r']:+.3f}",
                f"[{sharpe_color}]{m['sharpe']:.2f}[/]",
                f"{m['max_dd_r']:.2f}",
                pf_str,
                f"{m['cagr_pct']:+.1f}",
            )
        Console().print(t)


def run_walk_forward(
    dfs_or_df1h,
    df_15m:   pd.DataFrame | None = None,
    df_5m:    pd.DataFrame | None = None,
    params:   BacktestParams | None = None,
    is_start: str = "2019-01-01",
    is_end:   str = "2022-12-31",
    val_end:  str = "2023-12-31",
    oos_end:  str = "2025-12-31",
    # Extra TFs for Scenarios 2 & 3 (legacy positional call only)
    df_4h:    pd.DataFrame | None = None,
    df_1d:    pd.DataFrame | None = None,
) -> WalkForwardResult:
    """
    Run the backtest on three temporal splits.

    Calling conventions
    -------------------
    New (recommended):
        run_walk_forward(dfs, params, is_start, ...)
        where dfs = {"1H": df, "15M": df, "5M": df, "4H": df, "1D": df}

    Legacy:
        run_walk_forward(df_1h, df_15m, df_5m, params, ...)
        Extra TFs can be passed as df_4h= and df_1d= keyword args.

    Splits are applied to ALL DataFrames in the dict. No information leaks
    from later periods into earlier ones.
    """
    if params is None:
        params = BacktestParams()

    # ── Resolve to dfs dict ───────────────────────────────────────────────────
    if isinstance(dfs_or_df1h, dict):
        dfs = dfs_or_df1h
    else:
        dfs = {"1H": dfs_or_df1h, "15M": df_15m, "5M": df_5m}
        if df_4h is not None:
            dfs["4H"] = df_4h
        if df_1d is not None:
            dfs["1D"] = df_1d

    def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        idx = df.index
        if idx.tzinfo is None:
            idx = idx.tz_localize("UTC")
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
        return df[(idx >= s) & (idx < e)]

    is_dfs  = {tf: _slice(df, is_start, is_end)  for tf, df in dfs.items()}
    val_dfs = {tf: _slice(df, is_end,   val_end)  for tf, df in dfs.items()}
    oos_dfs = {tf: _slice(df, val_end,  oos_end)  for tf, df in dfs.items()}

    is_res  = run_single_backtest(is_dfs,  params=params)
    val_res = run_single_backtest(val_dfs, params=params)
    oos_res = run_single_backtest(oos_dfs, params=params)

    return WalkForwardResult(
        params      = params,
        is_result   = is_res,
        val_result  = val_res,
        oos_result  = oos_res,
        is_metrics  = compute_metrics(is_res),
        val_metrics = compute_metrics(val_res),
        oos_metrics = compute_metrics(oos_res),
    )


# ── Rolling walk-forward ──────────────────────────────────────────────────────

@dataclass
class RollingWindow:
    """Metrics for one IS → OOS pair in the rolling walk-forward."""
    is_start:    str
    is_end:      str
    oos_end:     str
    is_metrics:  dict
    oos_metrics: dict

    @property
    def is_r(self)       -> float: return self.is_metrics.get("total_r",  0.0)
    @property
    def oos_r(self)      -> float: return self.oos_metrics.get("total_r", 0.0)
    @property
    def is_sharpe(self)  -> float: return self.is_metrics.get("sharpe",   0.0)
    @property
    def oos_sharpe(self) -> float: return self.oos_metrics.get("sharpe",  0.0)
    @property
    def is_trades(self)  -> int:   return self.is_metrics.get("n_trades",  0)
    @property
    def oos_trades(self) -> int:   return self.oos_metrics.get("n_trades", 0)


def run_rolling_walk_forward(
    dfs:         dict,
    params:      BacktestParams | None = None,
    is_months:   int = 24,   # length of each IS window in months
    oos_months:  int = 6,    # length of each OOS window in months
    step_months: int = 6,    # step size between consecutive IS starts
) -> list[RollingWindow]:
    """
    Roll a fixed-width IS window across the full history; OOS follows each IS.

    Instead of a single 60/20/20 split, this repeatedly asks:
    "If I used these params on IS window X, how did they perform OOS?"

    Yields 6-10 windows over a 6-year dataset (2019-2025) with defaults.

    Parameters
    ----------
    dfs         : Full multi-TF dict from load_data().
    params      : Strategy parameters (BacktestParams).
    is_months   : Width of each IS window (default 24 = 2 years).
    oos_months  : Width of each OOS window (default 6 months).
    step_months : Advance between consecutive window starts (default 6).

    Returns
    -------
    List of RollingWindow, one per IS/OOS pair, chronological order.
    """
    if params is None:
        params = BacktestParams()

    tf_bias = params.tfs()[0]
    if tf_bias not in dfs:
        raise KeyError(
            f"Bias TF '{tf_bias}' not in dfs. Available: {list(dfs.keys())}"
        )

    ref     = dfs[tf_bias]
    t_start = ref.index[0]
    t_end   = ref.index[-1]
    if t_start.tzinfo is None:
        t_start = t_start.tz_localize("UTC")
    if t_end.tzinfo is None:
        t_end = t_end.tz_localize("UTC")

    def _slice(df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        idx = df.index if df.index.tzinfo else df.index.tz_localize("UTC")
        return df[(idx >= s) & (idx < e)]

    windows: list[RollingWindow] = []
    cur_is_start = t_start

    while True:
        cur_is_end  = cur_is_start + pd.DateOffset(months=is_months)
        cur_oos_end = cur_is_end   + pd.DateOffset(months=oos_months)

        if cur_is_end >= t_end:
            break                           # no room for a full IS window

        cur_oos_end = min(cur_oos_end, t_end)

        is_dfs  = {tf: _slice(df, cur_is_start, cur_is_end)  for tf, df in dfs.items()}
        oos_dfs = {tf: _slice(df, cur_is_end,   cur_oos_end) for tf, df in dfs.items()}

        is_res  = run_single_backtest(is_dfs,  params=params)
        oos_res = run_single_backtest(oos_dfs, params=params)

        windows.append(RollingWindow(
            is_start    = cur_is_start.strftime("%Y-%m-%d"),
            is_end      = cur_is_end.strftime("%Y-%m-%d"),
            oos_end     = cur_oos_end.strftime("%Y-%m-%d"),
            is_metrics  = compute_metrics(is_res),
            oos_metrics = compute_metrics(oos_res),
        ))

        if cur_oos_end >= t_end:
            break

        cur_is_start = cur_is_start + pd.DateOffset(months=step_months)

    return windows


# ── Optimiser ────────────────────────────────────────────────────────────────

PARAM_GRID: dict[str, list] = {
    "n_bias":    [5, 7, 9],
    "n_signal":  [3, 5, 7],
    "zone_lo":   [0.236, 0.382],
    "zone_hi":   [0.618, 0.786],
    "timeout_bars": [2, 4, 6],
}


def optimise(
    dfs_or_df1h,
    df_15m:     pd.DataFrame | None = None,
    df_5m:      pd.DataFrame | None = None,
    param_grid: dict[str, list] | None = None,
    is_start:   str  = "2019-01-01",
    is_end:     str  = "2022-12-31",
    min_trades: int  = 10,
    objective:  str  = "sharpe",    # "sharpe" | "profit_factor" | "total_r"
    verbose:    bool = True,
    # Fixed params that are NOT grid-searched (e.g. scenario)
    fixed_params: dict | None = None,
    # Legacy extra TFs
    df_4h: pd.DataFrame | None = None,
    df_1d: pd.DataFrame | None = None,
) -> tuple[BacktestParams, list[dict]]:
    """
    Grid-search on IN-SAMPLE data.

    Calling conventions: same dict-or-positional as run_walk_forward.

    Parameters
    ----------
    param_grid   : {param_name: [values]}.  Uses PARAM_GRID if None.
    fixed_params : Params held constant during the grid search (e.g. scenario=2).
    min_trades   : Discard results with fewer than this many trades.
    objective    : Metric to maximise.
    verbose      : Print progress.

    Returns
    -------
    (best_params, all_results_sorted_by_objective)
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    # Resolve dfs dict
    if isinstance(dfs_or_df1h, dict):
        dfs = dfs_or_df1h
    else:
        dfs = {"1H": dfs_or_df1h, "15M": df_15m, "5M": df_5m}
        if df_4h is not None:
            dfs["4H"] = df_4h
        if df_1d is not None:
            dfs["1D"] = df_1d

    def _slice(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index if df.index.tzinfo else df.index.tz_localize("UTC")
        s = pd.Timestamp(is_start, tz="UTC")
        e = pd.Timestamp(is_end,   tz="UTC") + pd.Timedelta(days=1)
        return df[(idx >= s) & (idx < e)]

    is_dfs = {tf: _slice(df) for tf, df in dfs.items()}

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total  = len(combos)

    base_kwargs = fixed_params or {}

    results = []

    for i, combo in enumerate(combos, 1):
        p = BacktestParams(**{**base_kwargs, **dict(zip(keys, combo))})
        if verbose and i % max(1, total // 10) == 0:
            print(f"  Optimiser {i}/{total} ({i/total*100:.0f}%) …")

        t0  = time.time()
        ex  = run_single_backtest(is_dfs, params=p)
        met = compute_metrics(ex)
        elapsed = time.time() - t0

        if met.get("n_trades", 0) < min_trades:
            continue

        results.append({
            "params":  p,
            "metrics": met,
            "elapsed": round(elapsed, 2),
        })

    if not results:
        return BacktestParams(**(base_kwargs or {})), []

    results.sort(key=lambda r: r["metrics"].get(objective, 0), reverse=True)
    best = results[0]["params"]
    return best, results
