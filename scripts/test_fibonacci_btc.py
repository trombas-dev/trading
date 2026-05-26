"""
test_fibonacci_btc.py — Fibonacci retracement + extension test on BTC-USD.

Finds the last CHoCH on the chosen timeframe, draws Fibonacci levels from
the swing that caused it, and generates a candlestick chart.

Usage:
    uv run python scripts/test_fibonacci_btc.py
    uv run python scripts/test_fibonacci_btc.py --tf 1h --n 5 --bars 120
    uv run python scripts/test_fibonacci_btc.py --tf 1d --n 7 --bars 200
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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hermes_trading.strategy.swing import find_swings, valid_n_values
from hermes_trading.strategy.structure import find_structure
from hermes_trading.strategy.fibonacci import (
    FibResult, FibLevel,
    calculate_fibs, fibs_from_last_choch,
)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (matches chart_structure_btc.py)
# ─────────────────────────────────────────────────────────────────────────────
BG           = "#0d1117"
GRID         = "#21262d"
BULL_CANDLE  = "#26a641"
BEAR_CANDLE  = "#da3633"

# Fib level colours (high → low: warm gradient + TP in cyan)
FIB_COLORS: dict[float, str] = {
    0.000: "#e6edf3",   # anchor top   — white
    0.236: "#ffe57f",   # 23.6 %       — light yellow
    0.382: "#ffa726",   # 38.2 %       — amber
    0.500: "#ef6c00",   # 50.0 %       — deep orange
    0.618: "#e53935",   # 61.8 %       — red (golden ratio)
    0.786: "#880e4f",   # 78.6 %       — dark magenta
    1.000: "#e6edf3",   # anchor bottom — white
    1.272: "#00bcd4",   # TP1 127.2 %  — cyan
    1.618: "#00e5ff",   # TP2 161.8 %  — bright cyan
}

ZONE_FILL_BULL = "#1b5e20"   # dark green shade for bullish golden zone
ZONE_FILL_BEAR = "#b71c1c"   # dark red  shade for bearish golden zone


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "1d":  ("1d",  "6y"),
    "1h":  ("1h",  "730d"),
    "4h":  ("1h",  "730d"),
    "15m": ("15m", "60d"),
}


def _fetch(ticker: str, tf: str) -> pd.DataFrame:
    yf_interval, yf_period = TIMEFRAME_MAP[tf]
    console.print(f"[dim]Downloading {ticker}  interval={yf_interval}  period={yf_period} ...[/]")
    raw = yf.download(ticker, interval=yf_interval, period=yf_period,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("yfinance returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    if tf == "4h":
        raw = (raw.resample("4h")
               .agg({"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"})
               .dropna())
    console.print(f"[dim]Got {len(raw):,} candles  ({raw.index[0].date()} - {raw.index[-1].date()})[/]")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Rich helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, style: str = "white") -> Panel:
    return Panel(f"[{style}]{value}[/]\n[dim]{label}[/]", expand=True)


def _levels_table(fib: FibResult) -> Table:
    t = Table(
        title=f"Fibonacci Levels  ({fib.direction.upper()})  "
              f"High=${fib.high:,.2f}  Low=${fib.low:,.2f}  "
              f"Range={fib.range_pct:.2f}%",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        title_style="bold",
    )
    t.add_column("Ratio",   justify="right")
    t.add_column("Label",   style="dim")
    t.add_column("Price",   justify="right")
    t.add_column("Dist from current", justify="right")
    t.add_column("Kind",    justify="center")

    # current close is the last price in the series — injected below
    return t


def _fill_levels_table(t: Table, fib: FibResult, current_price: float) -> None:
    for lv in fib.levels:
        dist = (current_price - lv.price) / lv.price * 100
        dist_text = Text(f"{dist:+.2f}%", style="green" if dist > 0 else "red")

        kind_style = {
            "anchor":      "dim",
            "retracement": "white",
            "extension":   "bold cyan",
        }.get(lv.kind, "white")

        t.add_row(
            f"{lv.ratio*100:.1f}%",
            lv.label,
            f"${lv.price:,.2f}",
            dist_text,
            Text(lv.kind, style=kind_style),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick renderer (shared with chart_structure_btc)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_candles(ax, df: pd.DataFrame) -> None:
    opens  = df["open"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    xs = np.arange(len(df))
    width = 0.6
    for i in range(len(df)):
        bull = closes[i] >= opens[i]
        col = BULL_CANDLE if bull else BEAR_CANDLE
        ax.plot([xs[i], xs[i]], [lows[i], highs[i]], color=col, linewidth=0.7, zorder=2)
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        if body_hi - body_lo < 1e-8:
            ax.plot([xs[i]-width/2, xs[i]+width/2], [closes[i], closes[i]],
                    color=col, linewidth=1, zorder=3)
        else:
            rect = mpatches.FancyBboxPatch(
                (xs[i]-width/2, body_lo), width, body_hi-body_lo,
                boxstyle="square,pad=0", facecolor=col, edgecolor="none", zorder=3)
            ax.add_patch(rect)


# ─────────────────────────────────────────────────────────────────────────────
# Main chart
# ─────────────────────────────────────────────────────────────────────────────

def _save_chart(df_full: pd.DataFrame, fib: FibResult,
                n: int, tf: str, n_bars: int, out_path: Path) -> None:
    try:
        import matplotlib.ticker
    except ImportError:
        console.print("[yellow]matplotlib not available[/]")
        return

    matplotlib.rcParams.update({"font.family": "monospace", "font.size": 9})

    # ── Slice view window ─────────────────────────────────────────────────────
    df = df_full.iloc[-n_bars:].copy()
    bar_offset = len(df_full) - n_bars

    fig, ax = plt.subplots(figsize=(22, 11), facecolor=BG)
    ax.set_facecolor(BG)

    # ── Candlesticks ──────────────────────────────────────────────────────────
    _draw_candles(ax, df)

    price_min = df["low"].min()
    price_max = df["high"].max()
    visible_min = min(price_min, fib.tp2.price if fib.tp2 else fib.low) * 0.995
    visible_max = max(price_max, fib.tp2.price if fib.tp2 else fib.high) * 1.005

    # ── Golden zone shaded band ───────────────────────────────────────────────
    zone_lo, zone_hi = fib.zone(0.382, 0.618)
    zone_col = ZONE_FILL_BULL if fib.direction == "bullish" else ZONE_FILL_BEAR
    ax.axhspan(zone_lo, zone_hi, color=zone_col, alpha=0.18, zorder=1,
               label="Golden zone 38.2-61.8%")

    # ── Extended zone shaded band (0.618-0.786) ───────────────────────────────
    ext_lo, ext_hi = fib.zone(0.618, 0.786)
    ax.axhspan(ext_lo, ext_hi, color=zone_col, alpha=0.09, zorder=1,
               label="Deep zone 61.8-78.6%")

    # ── Fibonacci level lines ─────────────────────────────────────────────────
    # Find x position of the swing high and swing low inside the view
    sh_xi = fib.swing_high.index - bar_offset
    sl_xi = fib.swing_low.index  - bar_offset
    x_anchor = max(0, min(sh_xi, sl_xi))   # leftmost swing in view (or 0)
    x_end    = len(df) - 1

    for lv in fib.levels:
        color = FIB_COLORS.get(lv.ratio, "#888888")
        lw    = 1.4 if lv.kind == "extension" else 0.9
        ls    = "--" if lv.kind == "extension" else "-"
        alpha = 0.85 if lv.kind != "anchor" else 0.5

        # Draw the line across the chart (from x_anchor onward)
        x_start = max(0, x_anchor)
        ax.hlines(lv.price, x_start, x_end,
                  colors=color, linewidth=lw, linestyles=ls,
                  alpha=alpha, zorder=4)

        # Price + label on the RIGHT side
        ax.annotate(
            f" {lv.label}   ${lv.price:,.0f}",
            xy=(x_end, lv.price),
            fontsize=7.5,
            color=color,
            va="center",
            ha="left",
            zorder=7,
            annotation_clip=False,
        )

    # ── Mark swing high and swing low ────────────────────────────────────────
    if 0 <= sh_xi < len(df):
        ax.scatter(sh_xi, fib.swing_high.price,
                   color=FIB_COLORS[0.0], marker="^", s=120, zorder=8,
                   label=f"Swing High ${fib.high:,.0f}")
    if 0 <= sl_xi < len(df):
        ax.scatter(sl_xi, fib.swing_low.price,
                   color=FIB_COLORS[1.0], marker="v", s=120, zorder=8,
                   label=f"Swing Low  ${fib.low:,.0f}")

    # ── Current price line ────────────────────────────────────────────────────
    current_price = df["close"].iloc[-1]
    ax.axhline(current_price, color="#8b949e", linewidth=0.7,
               linestyle=":", zorder=5, label=f"Current ${current_price:,.0f}")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_ylim(visible_min, visible_max)
    ax.set_xlim(-1, len(df) + 15)   # +15 leaves room for right-side labels

    step = max(1, len(df) // 18)
    tick_pos    = list(range(0, len(df), step))
    tick_labels = [str(df.index[i])[:13] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right",
                       color="#8b949e", fontsize=8)
    ax.yaxis.set_tick_params(colors="#8b949e")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Price (USD)", color="#8b949e")
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # ── Title ─────────────────────────────────────────────────────────────────
    nearest = fib.nearest_level(current_price)
    ax.set_title(
        f"BTC-USD ({tf})  —  Fibonacci Levels  (N={n}, {fib.direction.upper()})  "
        f"[{df.index[0].date()} - {df.index[-1].date()}]  |  "
        f"Range: {fib.range_pct:.2f}%  |  "
        f"Price near: {nearest.label}  |  "
        f"TP1: ${fib.tp1.price:,.0f}  TP2: ${fib.tp2.price:,.0f}",
        color="#e6edf3", fontsize=10, pad=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=zone_col,  alpha=0.35, label="Golden zone 38.2-61.8%"),
        mpatches.Patch(facecolor=zone_col,  alpha=0.18, label="Deep zone 61.8-78.6%"),
        mpatches.Patch(facecolor=BULL_CANDLE, label="Bull candle"),
        mpatches.Patch(facecolor=BEAR_CANDLE, label="Bear candle"),
    ]
    # Add coloured lines for TP1 / TP2
    legend_patches += [
        Line2D([0],[0], color=FIB_COLORS[1.272], lw=1.5, ls="--", label="TP1 127.2%"),
        Line2D([0],[0], color=FIB_COLORS[1.618], lw=1.5, ls="--", label="TP2 161.8%"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", ncol=3)

    plt.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Chart saved -> {out_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Fibonacci test on BTC-USD")
    parser.add_argument("--n",    type=int, default=7, choices=valid_n_values())
    parser.add_argument("--tf",   type=str, default="1d", choices=list(TIMEFRAME_MAP))
    parser.add_argument("--bars", type=int, default=200,
                        help="Bars to show in chart (default 200)")
    args = parser.parse_args()

    console.print()
    console.print("[bold white]Hermes Trading — Step 3: Fibonacci Levels[/]")
    console.print(f"[dim]Timeframe: {args.tf}  |  N: {args.n}  |  Chart bars: {args.bars}[/]")
    console.print()

    # ── Fetch + detect ────────────────────────────────────────────────────────
    df = _fetch("BTC-USD", tf=args.tf)
    swing_res  = find_swings(df, n=args.n)
    struct_res = find_structure(df, swing_res)

    # ── Build Fibonacci from last CHoCH ───────────────────────────────────────
    fib = fibs_from_last_choch(struct_res, swing_res)
    if fib is None:
        console.print("[red]No CHoCH found — cannot draw Fibonacci levels.[/]")
        return

    current_price = float(df["close"].iloc[-1])
    last_choch    = struct_res.last_choch()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    in_golden = fib.is_in_zone(current_price, 0.382, 0.618)
    in_deep   = fib.is_in_zone(current_price, 0.618, 0.786)
    zone_txt  = "GOLDEN ZONE" if in_golden else ("DEEP ZONE" if in_deep else "outside zone")
    zone_style= "bold green" if in_golden else ("bold yellow" if in_deep else "dim")
    bias_style= "green" if fib.direction == "bullish" else "red"
    nearest   = fib.nearest_level(current_price)

    console.print(
        Columns([
            _kpi("Direction",     fib.direction.upper(), bias_style),
            _kpi("Swing High",    f"${fib.high:,.2f}"),
            _kpi("Swing Low",     f"${fib.low:,.2f}"),
            _kpi("Range",         f"{fib.range_pct:.2f}%"),
            _kpi("Current price", f"${current_price:,.2f}"),
            _kpi("Nearest level", nearest.label.strip()),
            _kpi("Zone status",   zone_txt, zone_style),
        ], equal=True)
    )
    console.print()

    # ── TP summary ────────────────────────────────────────────────────────────
    tp1 = fib.tp1
    tp2 = fib.tp2
    if tp1 and tp2:
        dist_tp1 = (tp1.price - current_price) / current_price * 100
        dist_tp2 = (tp2.price - current_price) / current_price * 100
        console.print(
            Columns([
                _kpi("TP1 (127.2%)", f"${tp1.price:,.2f}", "cyan"),
                _kpi("TP1 distance", f"{dist_tp1:+.2f}%",  "cyan"),
                _kpi("TP2 (161.8%)", f"${tp2.price:,.2f}", "bold cyan"),
                _kpi("TP2 distance", f"{dist_tp2:+.2f}%",  "bold cyan"),
            ], equal=True)
        )
        console.print()

    # ── CHoCH context ─────────────────────────────────────────────────────────
    console.print(f"[dim]Last CHoCH:[/]  {last_choch}")
    console.print(
        f"[dim]Swing High bar {fib.swing_high.index} "
        f"({str(fib.swing_high.timestamp)[:10]})  "
        f"Swing Low bar {fib.swing_low.index} "
        f"({str(fib.swing_low.timestamp)[:10]})[/]"
    )
    console.print()

    # ── Levels table ──────────────────────────────────────────────────────────
    t = _levels_table(fib)
    _fill_levels_table(t, fib, current_price)
    console.print(t)
    console.print()

    # ── Zone check ────────────────────────────────────────────────────────────
    zone_lo, zone_hi = fib.zone(0.382, 0.618)
    console.print(
        f"[dim]Golden zone (38.2-61.8%):[/]  "
        f"[yellow]${zone_lo:,.2f}[/] — [yellow]${zone_hi:,.2f}[/]  "
        f"Current price {'[bold green]INSIDE[/]' if in_golden else '[dim]outside[/]'}"
    )
    deep_lo, deep_hi = fib.zone(0.618, 0.786)
    console.print(
        f"[dim]Deep zone    (61.8-78.6%):[/]  "
        f"[yellow]${deep_lo:,.2f}[/] — [yellow]${deep_hi:,.2f}[/]  "
        f"Current price {'[bold yellow]INSIDE[/]' if in_deep else '[dim]outside[/]'}"
    )
    console.print()

    # ── Chart ─────────────────────────────────────────────────────────────────
    out = (Path(__file__).parent.parent / "charts"
           / f"fibonacci_{args.tf}_n{args.n}.png")
    _save_chart(df, fib, n=args.n, tf=args.tf,
                n_bars=min(args.bars, len(df)), out_path=out)

    console.print(f"[bold green]OK[/]  {fib.summary()}")
    console.print()


if __name__ == "__main__":
    main()
