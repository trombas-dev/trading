"""
run_backtest.py -- CLI runner for the Hermes MTF walk-forward backtest.

Modes
-----
  backtest   Run the walk-forward backtest with fixed (or default) params.
  optimise   Grid-search on in-sample data, then evaluate best params on VAL + OOS.
  report     Load saved results from data/<SYMBOL>/backtest_result.parquet and print.

Data sources (tried in order)
-------------------------------
  1. Parquet files in data/<SYMBOL>/   (produced by download_mt5_data.py)
  2. yfinance live download             (max history limited -- good for quick tests)

Usage
-----
  # Walk-forward with defaults, BTC from yfinance
  uv run python scripts/run_backtest.py backtest --symbol BTCUSD --source yfinance

  # Walk-forward with custom params
  uv run python scripts/run_backtest.py backtest --symbol EURUSD --source yfinance \\
      --n-bias 7 --zone-lo 0.382 --zone-hi 0.786

  # Grid-search optimiser on in-sample, then walk-forward with winner
  uv run python scripts/run_backtest.py optimise --symbol BTCUSD --source yfinance

  # Print saved backtest result
  uv run python scripts/run_backtest.py report --symbol BTCUSD
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

# Project root on PYTHONPATH via uv
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# ── yfinance symbol map (same as downloader) ──────────────────────────────────
SYMBOL_MAP: dict[str, str] = {
    "BTCUSD":  "BTC-USD",
    "EURUSD":  "EURUSD=X",
    "GBPUSD":  "GBPUSD=X",
    "USDJPY":  "USDJPY=X",
    "XAUUSD":  "GC=F",
    "GBPJPY":  "GBPJPY=X",
    "EURJPY":  "EURJPY=X",
    "NZDUSD":  "NZDUSD=X",
    "EURGBP":  "EURGBP=X",
    "USDCHF":  "USDCHF=X",
    "GBPCHF":  "GBPCHF=X",
    "CHFJPY":  "CHFJPY=X",
    "GBPAUD":  "GBPAUD=X",
}

# ── Spread helpers ────────────────────────────────────────────────────────────
# Converts spread_pts (round-trip, instrument points) -> half_spread (price units).
# half_spread is deducted once on entry, once on exit inside the backtest engine.
_POINT_SIZE: dict[str, float] = {
    # Forex majors / minors (5-digit, point = 0.00001)
    "EURUSD": 0.00001, "GBPUSD": 0.00001, "NZDUSD": 0.00001,
    "USDCHF": 0.00001, "EURGBP": 0.00001, "GBPAUD": 0.00001, "GBPCHF": 0.00001,
    "AUDUSD": 0.00001, "USDCAD": 0.00001,
    # JPY pairs (3-digit, point = 0.001)
    "USDJPY": 0.001,   "EURJPY": 0.001,   "GBPJPY": 0.001,   "CHFJPY": 0.001,
    # Commodities / crypto (point = 0.01)
    "XAUUSD": 0.01,    "BTCUSD": 0.01,
}
# Live bid-ask spreads measured from MT5 (ICMarkets EU Demo, 2026-05-26)
# Spread = (ask - bid) / point_size  — raw market cost, no extra buffer.
_SPREAD_PTS: dict[str, int] = {
    "EURUSD":  9,  "GBPUSD": 12,  "NZDUSD": 11,
    "USDCHF":  9,  "EURGBP": 13,  "GBPAUD": 18,  "GBPCHF": 15,
    "AUDUSD": 10,  "USDCAD": 10,
    "USDJPY": 10,  "EURJPY": 21,  "GBPJPY": 22,  "CHFJPY": 16,
    "XAUUSD": 19,  "BTCUSD": 1200,
}

def _default_half_spread(symbol: str) -> float:
    """Return the default half-spread for a symbol (from ICMarkets typical spreads)."""
    pts = _SPREAD_PTS.get(symbol, 20)
    ps  = _POINT_SIZE.get(symbol, 0.00001)
    return pts * ps / 2


# ── Data loaders ──────────────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lower-case columns, UTC DatetimeIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    return df


ALL_TFS = ("1D", "4H", "1H", "15M", "5M")


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


def _load_parquet(symbol: str) -> dict[str, pd.DataFrame]:
    """Load all available TF parquet files for a symbol."""
    sym_dir = DATA_DIR / symbol
    dfs: dict[str, pd.DataFrame] = {}
    for tf in ALL_TFS:
        path = sym_dir / f"{tf}.parquet"
        if path.exists():
            dfs[tf] = _normalise(pd.read_parquet(path, engine="pyarrow"))
    required = {"1H", "15M", "5M"}
    missing = required - dfs.keys()
    if missing:
        raise FileNotFoundError(
            f"Parquet files missing for {symbol}: {sorted(missing)}\n"
            f"Run:  uv run python scripts/download_mt5_data.py --symbol {symbol}"
        )
    return dfs


def _load_yfinance(symbol: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    ticker = SYMBOL_MAP.get(symbol)
    if not ticker:
        raise ValueError(f"No yfinance ticker for '{symbol}'")

    print(f"  Downloading {symbol} ({ticker}) from yfinance …")

    def _dl(interval: str, period: str) -> pd.DataFrame:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError(f"yfinance returned empty data for {ticker} / {interval}")
        return _normalise(raw)

    df_1h  = _dl("1h",  "730d")
    df_15m = _dl("15m", "60d")
    df_5m  = _dl("5m",  "60d")

    # Derive 4H and 1D from 1H via resampling
    df_4h = _resample(df_1h, "4h")
    df_1d = _resample(df_1h, "1D")

    # Align 15M/5M to start at earliest shared point
    start = df_15m.index[0]
    df_1h = df_1h[df_1h.index >= start]
    df_4h = df_4h[df_4h.index >= start]
    df_1d = df_1d[df_1d.index >= start]

    return {"1D": df_1d, "4H": df_4h, "1H": df_1h, "15M": df_15m, "5M": df_5m}


def load_data(symbol: str, source: str) -> dict[str, pd.DataFrame]:
    if source == "parquet":
        return _load_parquet(symbol)
    return _load_yfinance(symbol)


# ── Auto-split helper ─────────────────────────────────────────────────────────

def _resolve_splits(
    dfs:      dict,
    is_start: str,
    is_end:   str,
    val_end:  str,
    oos_end:  str,
) -> tuple[str, str, str, str]:
    """
    Return (is_start, is_end, val_end, oos_end) as date strings.

    If the requested IS window falls entirely outside the available data
    (common when using yfinance which only covers ~730 days of 1H),
    auto-compute splits from the actual shared date range using a
    60 / 20 / 20 split.
    """
    # Actual shared range across all available timeframes
    t_start = max(df.index[0]  for df in dfs.values())
    t_end   = min(df.index[-1] for df in dfs.values())

    req_is_start = pd.Timestamp(is_start, tz="UTC")
    req_is_end   = pd.Timestamp(is_end,   tz="UTC")

    # Check if IS window is covered by data
    data_covers_is = (req_is_start <= t_end) and (req_is_end >= t_start)

    if data_covers_is:
        return is_start, is_end, val_end, oos_end

    # Auto-split: 60 % IS / 20 % VAL / 20 % OOS
    span      = t_end - t_start
    is_cut    = t_start + span * 0.60
    val_cut   = t_start + span * 0.80

    auto_is_start = t_start.strftime("%Y-%m-%d")
    auto_is_end   = is_cut.strftime("%Y-%m-%d")
    auto_val_end  = val_cut.strftime("%Y-%m-%d")
    auto_oos_end  = t_end.strftime("%Y-%m-%d")

    print(
        f"  [auto-splits] requested IS ({is_start}->{is_end}) outside data range "
        f"({t_start.date()}->{t_end.date()}).\n"
        f"  Using auto 60/20/20 split:\n"
        f"    IS  : {auto_is_start} -> {auto_is_end}\n"
        f"    VAL : {auto_is_end}   -> {auto_val_end}\n"
        f"    OOS : {auto_val_end}  -> {auto_oos_end}\n"
    )
    return auto_is_start, auto_is_end, auto_val_end, auto_oos_end


# ── Report helpers ────────────────────────────────────────────────────────────

def _print_optimiser_table(results: list[dict], top_n: int = 10) -> None:
    from rich.table import Table
    from rich import box
    from rich.console import Console

    t = Table(
        title=f"Top {min(top_n, len(results))} parameter sets (in-sample)",
        box=box.SIMPLE_HEAD, show_edge=False,
    )
    cols = ["Rank", "n_bias", "n_sig", "zone_lo", "zone_hi",
            "timeout", "Trades", "WR%", "Sharpe", "PF", "Total R"]
    for c in cols:
        t.add_column(c, justify="right")

    for rank, r in enumerate(results[:top_n], 1):
        p = r["params"]
        m = r["metrics"]
        pf = m.get("profit_factor", 0)
        t.add_row(
            str(rank),
            str(p.n_bias),
            str(p.n_signal),
            f"{p.zone_lo:.3f}",
            f"{p.zone_hi:.3f}",
            str(p.timeout_bars),
            str(m.get("n_trades", 0)),
            f"{m.get('win_rate', 0):.1f}",
            f"{m.get('sharpe', 0):.2f}",
            f"{pf:.2f}" if pf != float('inf') else "inf",
            f"{m.get('total_r', 0):+.2f}",
        )
    Console().print(t)


def _print_equity_curve(curve: list[float], title: str = "Equity curve") -> None:
    """Print a tiny ASCII equity curve (no matplotlib needed)."""
    if not curve:
        print(f"  {title}: no data")
        return
    h = 8
    w = min(80, len(curve))
    step = max(1, len(curve) // w)
    sampled = curve[::step]
    mn, mx = min(sampled), max(sampled)
    span   = (mx - mn) or 1.0

    print(f"\n  {title}  (min={mn:+.2f}R  max={mx:+.2f}R  final={curve[-1]:+.2f}R)")
    for row in range(h, -1, -1):
        thresh = mn + row / h * span
        line   = ""
        for v in sampled:
            line += "*" if v >= thresh else " "
        label = f"{thresh:+6.2f}R |" if row % 4 == 0 else "        |"
        print(f"  {label}{line}")
    print(f"  {'':8s}+" + "-" * len(sampled))


# ── Sub-command: backtest ─────────────────────────────────────────────────────

def cmd_backtest(args: argparse.Namespace) -> None:
    from hermes_trading.strategy.backtest import (
        BacktestParams, run_walk_forward, run_rolling_walk_forward, SCENARIO_NAMES,
    )
    from rich.table import Table
    from rich import box
    from rich.console import Console

    print(f"\n[Backtest]  {args.symbol}  source={args.source}")
    print(f"  {SCENARIO_NAMES.get(args.scenario, f'Scenario {args.scenario}')}")
    dfs = load_data(args.symbol, args.source)
    for tf, df in sorted(dfs.items()):
        print(f"  {tf:4s} bars: {len(df):,}")

    half_spread = (args.spread_pts * _POINT_SIZE.get(args.symbol, 0.00001) / 2
                   if args.spread_pts is not None
                   else _default_half_spread(args.symbol))

    params = BacktestParams(
        n_bias            = args.n_bias,
        n_signal          = args.n_signal,
        n_entry           = args.n_entry,
        zone_lo           = args.zone_lo,
        zone_hi           = args.zone_hi,
        timeout_bars      = args.timeout_bars,
        max_bars_in_trade = args.max_bars,
        scenario          = args.scenario,
        use_bos_windows   = args.use_bos_windows,
        bos_fib_mode      = args.bos_fib_mode,
        half_spread       = half_spread,
        regime_filter     = args.regime_filter,
        regime_tf         = args.regime_tf,
        regime_adx_period = args.regime_adx_period,
        regime_adx_min    = args.regime_adx_min,
        regime_atr_short  = args.regime_atr_short,
        regime_atr_long   = args.regime_atr_long,
        regime_atr_ratio  = args.regime_atr_ratio,
    )
    print(f"  params  : {params.label()}")
    pts = args.spread_pts if args.spread_pts is not None else _SPREAD_PTS.get(args.symbol, 20)
    print(f"  spread  : {pts} pts  (half_spread={half_spread:.6f})\n")

    # ── Rolling walk-forward mode ─────────────────────────────────────────────
    if args.rolling:
        print(
            f"[Rolling walk-forward]  "
            f"IS={args.rolling_is_months}mo  "
            f"OOS={args.rolling_oos_months}mo  "
            f"step={args.rolling_step_months}mo\n"
        )
        windows = run_rolling_walk_forward(
            dfs,
            params      = params,
            is_months   = args.rolling_is_months,
            oos_months  = args.rolling_oos_months,
            step_months = args.rolling_step_months,
        )

        if not windows:
            print("  No windows produced — dataset too short for these settings.")
            return

        t = Table(
            title=f"{args.symbol} — Rolling Walk-Forward ({len(windows)} windows)",
            box=box.SIMPLE_HEAD, show_edge=False,
        )
        for col in ["IS start", "IS end", "OOS end",
                    "IS trades", "IS R", "IS Sharpe",
                    "OOS trades", "OOS R", "OOS Sharpe"]:
            t.add_column(col, justify="right")

        total_oos_r = 0.0
        pos_windows = 0
        for w in windows:
            oos_r = w.oos_r
            total_oos_r += oos_r
            pos_windows += int(oos_r > 0)
            clr = "green" if oos_r > 0 else "red"
            t.add_row(
                w.is_start, w.is_end, w.oos_end,
                str(w.is_trades),
                f"{w.is_r:+.2f}",
                f"{w.is_sharpe:.2f}",
                str(w.oos_trades),
                f"[{clr}]{oos_r:+.2f}[/]",
                f"{w.oos_sharpe:.2f}",
            )

        Console().print(t)
        print(
            f"\n  OOS positive windows: {pos_windows}/{len(windows)}  "
            f"Total OOS R: {total_oos_r:+.2f}  "
            f"Avg OOS R/window: {total_oos_r/len(windows):+.2f}"
        )
        return

    # ── Single walk-forward mode (default) ────────────────────────────────────
    is_start, is_end, val_end, oos_end = _resolve_splits(
        dfs,
        args.is_start, args.is_end, args.val_end, args.oos_end,
    )

    result = run_walk_forward(
        dfs,
        params   = params,
        is_start = is_start,
        is_end   = is_end,
        val_end  = val_end,
        oos_end  = oos_end,
    )
    result.print_summary()

    # Equity curves
    for label, m in [("In-sample",     result.is_metrics),
                     ("Validation",    result.val_metrics),
                     ("Out-of-sample", result.oos_metrics)]:
        _print_equity_curve(m.get("equity_curve", []), title=label)

    # Optionally plot
    if args.chart:
        _plot_equity(result, args.symbol)


# ── Sub-command: optimise ─────────────────────────────────────────────────────

def cmd_optimise(args: argparse.Namespace) -> None:
    from hermes_trading.strategy.backtest import (
        optimise, run_walk_forward,
    )

    print(f"\n[Optimise]  {args.symbol}  source={args.source}  scenario={args.scenario}")
    dfs = load_data(args.symbol, args.source)
    for tf, df in sorted(dfs.items()):
        print(f"  {tf:4s} bars: {len(df):,}")
    print()

    is_start, is_end, val_end, oos_end = _resolve_splits(
        dfs,
        args.is_start, args.is_end, args.val_end, args.oos_end,
    )

    hs = (args.spread_pts * _POINT_SIZE.get(args.symbol, 0.00001) / 2
          if args.spread_pts is not None
          else _default_half_spread(args.symbol))
    pts = args.spread_pts if args.spread_pts is not None else _SPREAD_PTS.get(args.symbol, 20)
    print(f"  spread  : {pts} pts  (half_spread={hs:.6f})\n")

    best_params, all_results = optimise(
        dfs,
        is_start      = is_start,
        is_end        = is_end,
        min_trades    = args.min_trades,
        objective     = args.objective,
        verbose       = True,
        fixed_params  = {
            "scenario":          args.scenario,
            "use_bos_windows":   args.use_bos_windows,
            "bos_fib_mode":      args.bos_fib_mode,
            "half_spread":       hs,
            "regime_filter":     args.regime_filter,
            "regime_tf":         args.regime_tf,
            "regime_adx_period": args.regime_adx_period,
            "regime_adx_min":    args.regime_adx_min,
            "regime_atr_short":  args.regime_atr_short,
            "regime_atr_long":   args.regime_atr_long,
            "regime_atr_ratio":  args.regime_atr_ratio,
        },
    )

    if not all_results:
        print("[WARN] No parameter combination produced enough trades. Try lowering --min-trades.")
        return

    print(f"\n  Grid search complete. {len(all_results)} valid combos found.")
    print(f"  Best params: {best_params.label()}\n")
    _print_optimiser_table(all_results, top_n=10)

    # Walk-forward with winner
    print("\n[Walk-forward with best params]")
    result = run_walk_forward(
        dfs,
        params   = best_params,
        is_start = is_start,
        is_end   = is_end,
        val_end  = val_end,
        oos_end  = oos_end,
    )
    result.print_summary()

    for label, m in [("In-sample",     result.is_metrics),
                     ("Validation",    result.val_metrics),
                     ("Out-of-sample", result.oos_metrics)]:
        _print_equity_curve(m.get("equity_curve", []), title=label)

    if args.chart:
        _plot_equity(result, args.symbol)


# ── Sub-command: report ───────────────────────────────────────────────────────

def cmd_report(args: argparse.Namespace) -> None:
    """Quick-print a saved result (future: load from parquet)."""
    print(f"\n[Report]  {args.symbol}")
    sym_dir = DATA_DIR / args.symbol
    if not sym_dir.exists():
        print(f"  No data directory found for {args.symbol}. Run backtest first.")
        return

    files = list(sym_dir.glob("*.parquet"))
    print(f"  Found {len(files)} parquet file(s) in {sym_dir}:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<20s}  {size_kb:8.1f} KB")

    if not files:
        print("  Run:  uv run python scripts/download_mt5_data.py --symbol", args.symbol)


# ── Matplotlib equity plot ────────────────────────────────────────────────────

def _plot_equity(result, symbol: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed -- skipping chart.")
        return

    BG    = "#0d1117"
    GREEN = "#26a641"
    RED   = "#da3633"
    GOLD  = "#e3b341"
    BLUE  = "#388bfd"

    periods = [
        ("In-sample",     result.is_metrics,  GREEN),
        ("Validation",    result.val_metrics,  GOLD),
        ("Out-of-sample", result.oos_metrics,  BLUE),
    ]

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    ax  = fig.add_subplot(111, facecolor=BG)

    offset = 0.0
    for label, m, color in periods:
        curve = m.get("equity_curve", [])
        if not curve:
            continue
        x = list(range(len(curve)))
        y = [v + offset for v in curve]
        ax.plot(x, y, color=color, linewidth=1.5, label=f"{label} ({m.get('n_trades',0)} trades)")
        ax.axhline(offset, color=color, linewidth=0.4, linestyle="--", alpha=0.4)
        offset = y[-1]  # chain periods end-to-end

    ax.axhline(0, color="white", linewidth=0.6, alpha=0.3)
    ax.set_title(f"{symbol} — Walk-forward equity curve (R)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("white")
    ax.set_ylabel("Cumulative R", color="white")
    ax.legend(facecolor="#161b22", edgecolor="#444", labelcolor="white", fontsize=9)

    out = DATA_DIR / symbol / "equity_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"\n  Chart saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _add_date_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--is-start",  default="2019-01-01", help="In-sample start")
    p.add_argument("--is-end",    default="2022-12-31", help="In-sample end")
    p.add_argument("--val-end",   default="2023-12-31", help="Validation end")
    p.add_argument("--oos-end",   default="2025-12-31", help="OOS end")


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbol", default="BTCUSD",
                   help="Symbol to backtest (default: BTCUSD)")
    p.add_argument("--source", choices=["yfinance", "parquet"], default="yfinance",
                   help="Data source: yfinance (default) or parquet")


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Hermes MTF walk-forward backtest runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- backtest ---
    p_bt = sub.add_parser("backtest", help="Walk-forward backtest with fixed params")
    _add_data_args(p_bt)
    _add_date_args(p_bt)
    p_bt.add_argument("--scenario",        type=int,   default=1, choices=[1, 2, 3],
                      help="TF scenario: 1=1H/15M/5M (default), 2=4H/1H/15M, 3=1D/4H/1H")
    p_bt.add_argument("--use-bos-windows", action="store_true", default=False,
                      help="Also open trade windows on BOS events (default: CHoCH only)")
    p_bt.add_argument("--bos-fib-mode",
                      choices=["last_leg", "full_trend", "delayed"],
                      default="last_leg",
                      help="Fib calculation mode for BOS windows: "
                           "last_leg (default) | full_trend | delayed")
    p_bt.add_argument("--n-bias",      type=int,   default=7)
    p_bt.add_argument("--n-signal",    type=int,   default=5)
    p_bt.add_argument("--n-entry",     type=int,   default=3)
    p_bt.add_argument("--zone-lo",     type=float, default=0.382)
    p_bt.add_argument("--zone-hi",     type=float, default=0.786)
    p_bt.add_argument("--timeout-bars",type=int,   default=4)
    p_bt.add_argument("--max-bars",    type=int,   default=100)
    p_bt.add_argument("--chart",       action="store_true", help="Save equity chart PNG")
    # Rolling walk-forward
    p_bt.add_argument("--rolling",     action="store_true", default=False,
                      help="Use rolling walk-forward instead of single 60/20/20 split")
    p_bt.add_argument("--rolling-is-months",   type=int, default=24,
                      help="IS window length in months (default: 24)")
    p_bt.add_argument("--rolling-oos-months",  type=int, default=6,
                      help="OOS window length in months (default: 6)")
    p_bt.add_argument("--rolling-step-months", type=int, default=6,
                      help="Step between IS window starts in months (default: 6)")
    # Regime filter
    p_bt.add_argument("--regime-filter",
                      choices=["none", "adx", "atr_ratio"], default="none",
                      help="Regime filter: none (default) | adx | atr_ratio")
    p_bt.add_argument("--regime-tf", default="1D",
                      help="Timeframe to compute regime indicator on (default: 1D)")
    p_bt.add_argument("--regime-adx-period", type=int,   default=14,
                      help="ADX period (default: 14)")
    p_bt.add_argument("--regime-adx-min",    type=float, default=20.0,
                      help="Min ADX to allow trading (default: 20.0)")
    p_bt.add_argument("--regime-atr-short",  type=int,   default=10,
                      help="Short ATR period for atr_ratio filter (default: 10)")
    p_bt.add_argument("--regime-atr-long",   type=int,   default=50,
                      help="Long ATR period for atr_ratio filter (default: 50)")
    p_bt.add_argument("--regime-atr-ratio",  type=float, default=0.80,
                      help="Min ATR_short/ATR_long ratio (default: 0.80)")
    p_bt.add_argument("--spread-pts", type=int, default=None, metavar="PTS",
                      help="Round-trip spread in instrument points (default: per-symbol table). "
                           "E.g. EURUSD=20, USDJPY=30, XAUUSD=30, BTCUSD=500. "
                           "Pass 0 to disable spread.")

    # --- optimise ---
    p_op = sub.add_parser("optimise", help="Grid-search IS then walk-forward with winner")
    _add_data_args(p_op)
    _add_date_args(p_op)
    p_op.add_argument("--scenario",        type=int,   default=1, choices=[1, 2, 3],
                      help="TF scenario: 1=1H/15M/5M (default), 2=4H/1H/15M, 3=1D/4H/1H")
    p_op.add_argument("--use-bos-windows", action="store_true", default=False,
                      help="Also open trade windows on BOS events (default: CHoCH only)")
    p_op.add_argument("--bos-fib-mode",
                      choices=["last_leg", "full_trend", "delayed"],
                      default="last_leg",
                      help="Fib calculation mode for BOS windows: "
                           "last_leg (default) | full_trend | delayed")
    p_op.add_argument("--min-trades",  type=int,   default=10)
    p_op.add_argument("--objective",
                      choices=["sharpe", "profit_factor", "total_r"],
                      default="sharpe")
    p_op.add_argument("--chart",       action="store_true", help="Save equity chart PNG")
    # Regime filter (same as backtest)
    p_op.add_argument("--regime-filter",
                      choices=["none", "adx", "atr_ratio"], default="none",
                      help="Regime filter: none (default) | adx | atr_ratio")
    p_op.add_argument("--regime-tf", default="1D",
                      help="Timeframe to compute regime indicator on (default: 1D)")
    p_op.add_argument("--regime-adx-period", type=int,   default=14)
    p_op.add_argument("--regime-adx-min",    type=float, default=20.0)
    p_op.add_argument("--regime-atr-short",  type=int,   default=10)
    p_op.add_argument("--regime-atr-long",   type=int,   default=50)
    p_op.add_argument("--regime-atr-ratio",  type=float, default=0.80)
    p_op.add_argument("--spread-pts", type=int, default=None, metavar="PTS",
                      help="Round-trip spread in instrument points (default: per-symbol table).")

    # --- report ---
    p_rp = sub.add_parser("report", help="Show saved backtest data for a symbol")
    p_rp.add_argument("--symbol", default="BTCUSD")

    args = parser.parse_args()

    if args.cmd == "backtest":
        cmd_backtest(args)
    elif args.cmd == "optimise":
        cmd_optimise(args)
    elif args.cmd == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
