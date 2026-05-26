"""
test_structure_btc.py — Visual test of the BOS/CHoCH structure classifier.

Downloads 6 years of daily BTC-USD, runs swing detection + structure
analysis, prints a rich report, and saves a matplotlib chart.

Usage:
    uv run python scripts/test_structure_btc.py
    uv run python scripts/test_structure_btc.py --n 5 --chart
    uv run python scripts/test_structure_btc.py --tf 1h --n 7 --chart
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hermes_trading.strategy.swing import find_swings, valid_n_values
from hermes_trading.strategy.structure import find_structure

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "1d":  ("1d",  "6y"),
    "1h":  ("1h",  "730d"),
    "4h":  ("1h",  "730d"),   # resample
    "15m": ("15m", "60d"),
}


def _fetch(ticker: str = "BTC-USD", tf: str = "1d") -> pd.DataFrame:
    yf_interval, yf_period = TIMEFRAME_MAP.get(tf, ("1d", "6y"))
    console.print(f"[dim]Downloading {ticker}  interval={yf_interval}  period={yf_period} ...[/]")

    raw = yf.download(ticker, interval=yf_interval, period=yf_period,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()

    if tf == "4h":
        raw = (
            raw.resample("4h")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna()
        )

    console.print(
        f"[dim]Got {len(raw):,} candles  "
        f"({raw.index[0].date()} - {raw.index[-1].date()})[/]"
    )
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Rich helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, style: str = "white") -> Panel:
    return Panel(f"[{style}]{value}[/]\n[dim]{label}[/]", expand=True)


def _events_table(events, title: str, n_rows: int = 25) -> Table:
    t = Table(title=title, box=box.SIMPLE_HEAD, show_edge=False, title_style="dim")
    t.add_column("Bar #",      justify="right",  style="dim")
    t.add_column("Date (UTC)", no_wrap=True)
    t.add_column("Kind",       justify="center")
    t.add_column("Direction",  justify="center")
    t.add_column("Level",      justify="right")
    t.add_column("Close",      justify="right")
    t.add_column("Delta %",    justify="right")

    for e in events[-n_rows:]:
        if e.direction == "bullish":
            dir_text = Text("BULL", style="green")
            lvl_color = "green"
        else:
            dir_text = Text("BEAR", style="red")
            lvl_color = "red"

        kind_text = Text(
            e.kind.upper(),
            style="bold cyan" if e.kind == "choch" else "white",
        )

        delta = (e.close - e.level) / e.level * 100
        delta_style = "green" if delta > 0 else "red"

        t.add_row(
            str(e.index),
            str(e.timestamp)[:16],
            kind_text,
            dir_text,
            f"${e.level:,.2f}",
            f"${e.close:,.2f}",
            Text(f"{delta:+.2f}%", style=delta_style),
        )
    return t


def _breakdown_table(struct_res, swing_res) -> Table:
    t = Table(
        title="Structure breakdown",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        title_style="bold",
    )
    t.add_column("Metric",  style="dim")
    t.add_column("Count",   justify="right")
    t.add_column("Notes",   style="dim")

    n_swings = len(swing_res.points)
    n_events = len(struct_res.events)
    n_bos    = len(struct_res.bos)
    n_choch  = len(struct_res.choch)
    n_bull   = len(struct_res.bullish_events)
    n_bear   = len(struct_res.bearish_events)

    t.add_row("Total swing points", str(n_swings), "highs + lows")
    t.add_row("Total structure events", str(n_events), "BOS + CHoCH")
    t.add_row("  BOS events",  str(n_bos),   "trend continuation")
    t.add_row("  CHoCH events", str(n_choch), "trend reversal signal")
    t.add_row("  Bullish events", str(n_bull), "")
    t.add_row("  Bearish events", str(n_bear), "")

    if n_events > 1:
        idxs = [e.index for e in struct_res.events]
        diffs = [b - a for a, b in zip(idxs, idxs[1:])]
        t.add_row(
            "Avg bars between events",
            f"{np.mean(diffs):.1f}",
            f"min={min(diffs)}  max={max(diffs)}",
        )
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def _save_chart(df: pd.DataFrame, swing_res, struct_res, n: int,
                tf: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.patches as mpatches
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping chart[/]")
        return

    fig, ax = plt.subplots(figsize=(20, 8))

    # ── Price line ────────────────────────────────────────────────────────────
    ax.plot(df.index, df["close"], color="#555555", linewidth=0.7,
            label="Close", zorder=1)

    # ── Swing points (small, behind structure) ────────────────────────────────
    if swing_res.highs:
        h_t = [df.index[p.index] for p in swing_res.highs]
        h_p = [p.price for p in swing_res.highs]
        ax.scatter(h_t, h_p, color="#4caf50", marker="^", s=20,
                   alpha=0.5, zorder=2, label=f"Swing High (N={n})")
    if swing_res.lows:
        l_t = [df.index[p.index] for p in swing_res.lows]
        l_p = [p.price for p in swing_res.lows]
        ax.scatter(l_t, l_p, color="#f44336", marker="v", s=20,
                   alpha=0.5, zorder=2, label=f"Swing Low (N={n})")

    # ── Structure events (larger, prominent) ──────────────────────────────────
    bull_bos   = [e for e in struct_res.events if e.kind == "bos"   and e.direction == "bullish"]
    bear_bos   = [e for e in struct_res.events if e.kind == "bos"   and e.direction == "bearish"]
    bull_choch = [e for e in struct_res.events if e.kind == "choch" and e.direction == "bullish"]
    bear_choch = [e for e in struct_res.events if e.kind == "choch" and e.direction == "bearish"]

    def _scatter_events(evts, color, marker, size, label, zorder=5):
        if evts:
            xs = [df.index[e.index] for e in evts]
            ys = [e.close for e in evts]
            ax.scatter(xs, ys, color=color, marker=marker, s=size,
                       zorder=zorder, label=label)

    _scatter_events(bull_bos,   "#00c853", "^", 80,  "BOS Bullish")
    _scatter_events(bear_bos,   "#d50000", "v", 80,  "BOS Bearish")
    _scatter_events(bull_choch, "#ff9800", "D", 100, "CHoCH Bullish")
    _scatter_events(bear_choch, "#aa00ff", "D", 100, "CHoCH Bearish")

    # ── Horizontal dashed lines for the 5 most recent events ─────────────────
    for e in struct_res.events[-5:]:
        color = "#00c853" if e.direction == "bullish" else "#d50000"
        ax.axhline(e.level, color=color, linewidth=0.6,
                   linestyle="--", alpha=0.4)

    ax.set_title(
        f"BTC-USD  ({tf})  —  BOS / CHoCH  (N={n}, close-based)  "
        f"[{df.index[0].date()} - {df.index[-1].date()}]",
        fontsize=12,
    )
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    console.print(f"[green]Chart saved -> {out_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Windows terminal encoding fix
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="BOS/CHoCH structure test on BTC-USD")
    parser.add_argument("--n",    type=int, default=7, choices=valid_n_values(),
                        help="N-candle window for swing detection (default 7)")
    parser.add_argument("--tf",   type=str, default="1d",
                        choices=list(TIMEFRAME_MAP),
                        help="Timeframe (default 1d)")
    parser.add_argument("--chart", action="store_true",
                        help="Save a matplotlib chart")
    args = parser.parse_args()

    console.print()
    console.print("[bold white]Hermes Trading — Step 2: BOS / CHoCH Structure Test[/]")
    console.print(f"[dim]Timeframe: {args.tf}  |  N: {args.n}  |  Break rule: CLOSE price[/]")
    console.print()

    df = _fetch("BTC-USD", tf=args.tf)

    # ── Run detectors ─────────────────────────────────────────────────────────
    swing_res  = find_swings(df, n=args.n)
    struct_res = find_structure(df, swing_res)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    final_bias = struct_res.bias()
    bias_style = "green" if final_bias == "bullish" else "red" if final_bias == "bearish" else "white"
    n_choch = len(struct_res.choch)
    choch_pct = f"{n_choch / len(struct_res.events) * 100:.1f}%" if struct_res.events else "0%"

    console.print(
        Columns([
            _kpi("Candles",       f"{len(df):,}"),
            _kpi("Swing points",  str(len(swing_res.points))),
            _kpi("Total events",  str(len(struct_res.events))),
            _kpi("BOS",           str(len(struct_res.bos))),
            _kpi("CHoCH",         str(n_choch), "cyan"),
            _kpi("CHoCH rate",    choch_pct, "cyan"),
            _kpi("Current bias",  final_bias.upper(), bias_style),
        ], equal=True)
    )
    console.print()

    # ── Breakdown table ───────────────────────────────────────────────────────
    console.print(_breakdown_table(struct_res, swing_res))
    console.print()

    # ── Last 25 events ────────────────────────────────────────────────────────
    console.print(_events_table(struct_res.events,
                                f"Last 25 structure events  (N={args.n}, close-based)"))
    console.print()

    # ── No-lookahead / mid-chart bias demo ────────────────────────────────────
    mid = len(df) // 2
    bias_at_mid = struct_res.bias(as_of=mid)
    evt_at_mid  = struct_res.last_event(as_of=mid)
    console.print(f"[dim]No-lookahead check — bias at bar {mid} "
                  f"({df.index[mid].date()}):[/]  "
                  f"[bold]{bias_at_mid.upper()}[/]")
    if evt_at_mid:
        console.print(f"  Last event as of bar {mid}: {evt_at_mid}")
    console.print()

    # ── CHoCH events only (reversal signals) ─────────────────────────────────
    if struct_res.choch:
        console.print(_events_table(struct_res.choch,
                                    "CHoCH events only (reversal signals)"))
        console.print()

    # ── Chart ─────────────────────────────────────────────────────────────────
    if args.chart:
        out = (Path(__file__).parent.parent / "charts"
               / f"structure_{args.tf}_n{args.n}.png")
        _save_chart(df, swing_res, struct_res, args.n, args.tf, out)

    console.print(f"[bold green]OK[/]  {struct_res.summary()}")
    console.print()


if __name__ == "__main__":
    main()
