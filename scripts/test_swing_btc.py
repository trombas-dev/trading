"""
test_swing_btc.py — Visual test of the swing detection module on BTC-USD data.

Downloads 6 years of daily BTC-USD via yfinance, runs find_swings() with
all valid N values, shows a rich table summary, and saves a matplotlib chart.

Usage:
    uv run python scripts/test_swing_btc.py
    uv run python scripts/test_swing_btc.py --n 7 --tf 1h --chart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make sure hermes_trading is importable from the project root ─────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hermes_trading.strategy.swing import find_swings, valid_n_values

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "1d": ("1d",  "6y"),   # daily   — 6 years
    "1h": ("1h",  "730d"), # hourly  — ~2 years (yfinance 1h limit)
    "4h": ("1h",  "730d"), # we resample 1h → 4h ourselves
    "15m": ("15m","60d"),  # 15-min  — yfinance limit ~60 days
}


def _fetch(ticker: str = "BTC-USD", tf: str = "1d") -> pd.DataFrame:
    yf_interval, yf_period = TIMEFRAME_MAP.get(tf, ("1d", "6y"))
    console.print(f"[dim]Downloading {ticker} interval={yf_interval} period={yf_period} …[/]")

    raw = yf.download(ticker, interval=yf_interval, period=yf_period, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yfinance ≥0.2 may return MultiIndex columns; flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()

    # Resample to 4H if requested
    if tf == "4h":
        raw = (
            raw.resample("4h")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna()
        )

    console.print(f"[dim]Got {len(raw):,} candles  "
                  f"({raw.index[0].date()} - {raw.index[-1].date()})[/]")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Rich table helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, style: str = "white") -> Panel:
    return Panel(f"[{style}]{value}[/]\n[dim]{label}[/]", expand=True)


def _n_summary_table(df: pd.DataFrame) -> Table:
    """One row per N value showing swing count and avg spacing."""
    t = Table(
        title="Swing Detection — all N values (daily BTC-USD)",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        title_style="bold",
    )
    t.add_column("N (candles)", justify="center")
    t.add_column("Lookback lb", justify="center")
    t.add_column("Swing Highs", justify="right")
    t.add_column("Swing Lows",  justify="right")
    t.add_column("Total",       justify="right")
    t.add_column("Avg spacing (candles)", justify="right", style="dim")

    for n in valid_n_values():
        res = find_swings(df, n=n)
        total = len(res.points)
        avg_sp = "-"
        if total > 1:
            idxs = [p.index for p in res.points]
            diffs = [b - a for a, b in zip(idxs, idxs[1:])]
            avg_sp = f"{np.mean(diffs):.1f}"
        def _cell(val: object, style: str) -> str:
            return f"[{style}]{val}[/]" if style else str(val)

        style = "bold cyan" if n == 7 else ""
        t.add_row(
            _cell(n,             style),
            _cell((n - 1) // 2,  style),
            _cell(len(res.highs),style),
            _cell(len(res.lows), style),
            _cell(total,         style),
            _cell(avg_sp,        style),
        )
    return t


def _points_table(res, title: str, kind: str, n_rows: int = 20) -> Table:
    t = Table(title=title, box=box.SIMPLE_HEAD, show_edge=False, title_style="dim")
    t.add_column("Bar #",       justify="right", style="dim")
    t.add_column("Confirmed at",justify="right", style="dim")
    t.add_column("Date (UTC)",  no_wrap=True)
    t.add_column("Price",       justify="right")

    pts = res.highs if kind == "high" else res.lows
    # Show last n_rows
    for p in pts[-n_rows:]:
        color = "green" if kind == "high" else "red"
        t.add_row(
            str(p.index),
            str(p.confirmed_at),
            str(p.timestamp)[:16],
            f"[{color}]${p.price:,.2f}[/]",
        )
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def _save_chart(df: pd.DataFrame, n: int, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping chart (pip install matplotlib)[/]")
        return

    res = find_swings(df, n=n)
    highs = res.highs
    lows  = res.lows

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(df.index, df["close"], color="#888888", linewidth=0.8, label="BTC-USD close")

    if highs:
        h_times  = [df.index[p.index] for p in highs]
        h_prices = [p.price for p in highs]
        ax.scatter(h_times, h_prices, color="#00c853", marker="^", s=40,
                   zorder=5, label=f"Swing High (N={n})")

    if lows:
        l_times  = [df.index[p.index] for p in lows]
        l_prices = [p.price for p in lows]
        ax.scatter(l_times, l_prices, color="#ff1744", marker="v", s=40,
                   zorder=5, label=f"Swing Low  (N={n})")

    ax.set_title(f"BTC-USD — Swing Highs & Lows  (N={n}, {len(highs)} highs, {len(lows)} lows)",
                 fontsize=13)
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    console.print(f"[green]Chart saved → {out_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Windows terminal encoding fix (cp1257 can't encode Unicode arrows etc.)
    import io
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Swing detection test on BTC-USD")
    parser.add_argument("--n",     type=int, default=7,
                        choices=valid_n_values(),
                        help="N-candle window for detailed table (default 7)")
    parser.add_argument("--tf",    type=str, default="1d",
                        choices=list(TIMEFRAME_MAP),
                        help="Timeframe: 1d | 1h | 4h | 15m (default 1d)")
    parser.add_argument("--chart", action="store_true",
                        help="Save a matplotlib chart (requires matplotlib)")
    args = parser.parse_args()

    console.print()
    console.print("[bold white]Hermes Trading — Step 1: Swing Detection Test[/]")
    console.print(f"[dim]Timeframe: {args.tf}  |  Detail N: {args.n}[/]")
    console.print()

    df = _fetch("BTC-USD", tf=args.tf)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    res_n = find_swings(df, n=args.n)
    lb    = (args.n - 1) // 2
    console.print(
        Columns([
            _kpi("Candles",      f"{len(df):,}"),
            _kpi("Date range",   f"{df.index[0].date()} - {df.index[-1].date()}"),
            _kpi("N (selected)", str(args.n)),
            _kpi("Lookback lb",  str(lb)),
            _kpi("Swing Highs",  str(len(res_n.highs)), "green"),
            _kpi("Swing Lows",   str(len(res_n.lows)),  "red"),
        ], equal=True)
    )
    console.print()

    # ── All-N summary ─────────────────────────────────────────────────────────
    if args.tf == "1d":
        # Only show the full N-comparison on daily (it's meaningful here)
        console.print(_n_summary_table(df))
        console.print()

    # ── Detailed tables for the chosen N ─────────────────────────────────────
    console.print(_points_table(res_n, f"Last 20 Swing HIGHs  (N={args.n})", "high"))
    console.print()
    console.print(_points_table(res_n, f"Last 20 Swing LOWs   (N={args.n})", "low"))
    console.print()

    # ── Confirmation-lag demo ─────────────────────────────────────────────────
    last_h = res_n.last_high()
    last_l = res_n.last_low()
    if last_h:
        console.print(f"  Last confirmed HIGH : {last_h}")
    if last_l:
        console.print(f"  Last confirmed LOW  : {last_l}")
    console.print()

    # ── Lookahead guard demo ──────────────────────────────────────────────────
    mid = len(df) // 2
    as_of_highs = res_n.last_n_highs(3, as_of=mid)
    console.print(f"[dim]Lookahead guard check — last 3 HIGHs confirmed as of bar {mid}:[/]")
    for p in as_of_highs:
        console.print(f"  {p}")
    console.print()

    # ── Optional chart ────────────────────────────────────────────────────────
    if args.chart:
        out = Path(__file__).parent.parent / "charts" / f"swing_{args.tf}_n{args.n}.png"
        _save_chart(df, args.n, out)

    console.print(f"[bold green]OK[/] — {res_n.summary()}")
    console.print()


if __name__ == "__main__":
    main()
