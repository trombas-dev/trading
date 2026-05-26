"""
test_entry_btc.py — Full MTF entry confirmation test on BTC-USD.

Pipeline:
  1H  N=7  -> swings -> structure -> fib (bias)
  15M N=5  -> swings -> structure -> signals  (zone + 15M struct)
  5M  N=3  -> swings -> structure -> entries  (5M struct confirmation)

Usage:
    uv run python scripts/test_entry_btc.py
    uv run python scripts/test_entry_btc.py --bars 250 --timeout 6
    uv run python scripts/test_entry_btc.py --zoom-entry 2  # zoom into entry #2
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
from hermes_trading.strategy.fibonacci import fibs_from_last_choch, FibResult
from hermes_trading.strategy.signal import find_signals, SignalResult
from hermes_trading.strategy.entry import find_entries, EntrySignal, EntryResult

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
BG           = "#0d1117"
GRID         = "#21262d"
BULL_CANDLE  = "#26a641"
BEAR_CANDLE  = "#da3633"
SIGNAL_COL   = "#ffeb3b"
ENTRY_COL    = "#00e5ff"
SL_COL       = "#f44336"
TP1_COL      = "#69f0ae"
TP2_COL      = "#b9f6ca"
ZONE_BULL    = "#1b5e20"
ZONE_BEAR    = "#b71c1c"

FIB_COLORS: dict[float, str] = {
    0.000: "#e6edf3", 0.236: "#ffe57f", 0.382: "#ffa726",
    0.500: "#ef6c00", 0.618: "#e53935", 0.786: "#880e4f",
    1.000: "#e6edf3", 1.272: "#00bcd4", 1.618: "#00e5ff",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(ticker: str, interval: str, period: str) -> pd.DataFrame:
    console.print(f"[dim]Downloading {ticker}  {interval}  {period} ...[/]")
    raw = yf.download(ticker, interval=interval, period=period,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"No data  {ticker} {interval}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index, utc=True)
    return raw[["open", "high", "low", "close", "volume"]].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Rich helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, style: str = "white") -> Panel:
    return Panel(f"[{style}]{value}[/]\n[dim]{label}[/]", expand=True)


def _entries_table(ent_res: EntryResult) -> Table:
    t = Table(
        title=f"Entry Signals  ({ent_res.bias}  timeout={ent_res.timeout_bars} 5M bars)",
        box=box.SIMPLE_HEAD, show_edge=False, title_style="bold cyan",
    )
    t.add_column("#",          justify="right", style="dim")
    t.add_column("Timestamp",  no_wrap=True)
    t.add_column("Entry $",    justify="right")
    t.add_column("SL $",       justify="right")
    t.add_column("TP1 $",      justify="right")
    t.add_column("TP2 $",      justify="right")
    t.add_column("R:R1",       justify="right")
    t.add_column("Fib level",  style="dim")
    t.add_column("5M confirm")

    for idx, e in enumerate(ent_res.entries, 1):
        rr_style = "bold green" if e.rr1 >= 2.0 else ("green" if e.rr1 >= 1.5 else "yellow")
        s = e.confirmed_by
        sc = "green" if s.direction == "bullish" else "red"
        t.add_row(
            str(idx),
            str(e.timestamp)[:16],
            f"[cyan]${e.entry_price:,.2f}[/]",
            f"[red]${e.stop_loss:,.2f}[/]",
            f"[green]${e.tp1:,.2f}[/]",
            f"[bold green]${e.tp2:,.2f}[/]",
            Text(f"{e.rr1:.2f}", style=rr_style),
            e.nearest_fib.label.strip(),
            f"[{sc}]{s.kind.upper()} {s.direction}[/]",
        )
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick renderer
# ─────────────────────────────────────────────────────────────────────────────

def _draw_candles(ax, df: pd.DataFrame) -> None:
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    w = 0.6
    for i in range(len(df)):
        col = BULL_CANDLE if c[i] >= o[i] else BEAR_CANDLE
        ax.plot([i, i], [l[i], h[i]], color=col, lw=0.7, zorder=2)
        lo, hi = min(o[i], c[i]), max(o[i], c[i])
        if hi - lo < 1e-8:
            ax.plot([i-w/2, i+w/2], [c[i], c[i]], color=col, lw=1, zorder=3)
        else:
            ax.add_patch(mpatches.FancyBboxPatch(
                (i-w/2, lo), w, hi-lo,
                boxstyle="square,pad=0", facecolor=col, edgecolor="none", zorder=3))


# ─────────────────────────────────────────────────────────────────────────────
# Overview chart  (5M last N bars, all entries)
# ─────────────────────────────────────────────────────────────────────────────

def _overview_chart(df_5m: pd.DataFrame, fib: FibResult,
                    sig_res: SignalResult, ent_res: EntryResult,
                    n_bars: int, out_path: Path) -> None:

    df   = df_5m.iloc[-n_bars:].copy()
    bar0 = len(df_5m) - n_bars

    matplotlib.rcParams.update({"font.family": "monospace", "font.size": 9})
    fig, (ax, ax_bot) = plt.subplots(
        2, 1, figsize=(22, 13), facecolor=BG,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05},
    )
    ax.set_facecolor(BG)
    ax_bot.set_facecolor(BG)

    _draw_candles(ax, df)

    # ── Fib zone bands ────────────────────────────────────────────────────────
    zone_col = ZONE_BULL if fib.direction == "bullish" else ZONE_BEAR
    glo, ghi = fib.zone(0.382, 0.618)
    dlo, dhi = fib.zone(0.618, 0.786)
    ax.axhspan(glo, ghi, color=zone_col, alpha=0.22, zorder=1)
    ax.axhspan(dlo, dhi, color=zone_col, alpha=0.12, zorder=1)

    # ── Fib levels ────────────────────────────────────────────────────────────
    for lv in fib.levels:
        col = FIB_COLORS.get(lv.ratio, "#888")
        lw  = 1.2 if lv.kind == "extension" else 0.7
        ls  = "--" if lv.kind == "extension" else "-"
        ax.axhline(lv.price, color=col, lw=lw, ls=ls, alpha=0.7, zorder=4)
        ax.annotate(f" {lv.ratio*100:.1f}%  ${lv.price:,.0f}",
                    xy=(len(df)-1, lv.price), fontsize=7, color=col,
                    va="center", ha="left", zorder=7, annotation_clip=False)

    # ── 15M signals (dashed vertical) ────────────────────────────────────────
    for s in sig_res.signals:
        # Find the nearest 5M bar to this 15M signal timestamp
        diffs = abs(df.index - s.timestamp)
        xi = diffs.argmin() if len(diffs) > 0 else -1
        if 0 <= xi < len(df):
            ax.axvline(xi, color=SIGNAL_COL, lw=0.6, ls="--",
                       alpha=0.35, zorder=5)

    # ── Entry signals ─────────────────────────────────────────────────────────
    for n, e in enumerate(ent_res.entries, 1):
        xi = e.index - bar0
        if xi < 0 or xi >= len(df):
            continue

        # Entry star
        ax.scatter(xi, e.entry_price, color=ENTRY_COL, marker="*",
                   s=260, zorder=9, linewidths=0.5, edgecolors="#000")

        # SL / TP horizontal snippets (20 bars wide)
        x_end = min(xi + 25, len(df) - 1)
        ax.hlines(e.stop_loss, xi, x_end, colors=SL_COL,  lw=1.2, ls="-",  zorder=6)
        ax.hlines(e.tp1,       xi, x_end, colors=TP1_COL, lw=1.2, ls="-",  zorder=6)
        ax.hlines(e.tp2,       xi, x_end, colors=TP2_COL, lw=1.0, ls="--", zorder=6)

        # Label
        ax.annotate(
            f" #{n}\n ${e.entry_price:,.0f}\n R:R {e.rr1:.1f}",
            xy=(xi, e.entry_price),
            xytext=(xi + 1,
                    e.entry_price * (1.0018 if e.direction == "bullish" else 0.9982)),
            fontsize=7.5, color=ENTRY_COL, va="bottom", zorder=10,
            annotation_clip=True,
        )

    # ── Bottom status bar ─────────────────────────────────────────────────────
    in_z = False
    for i in range(len(df)):
        abs_i = bar0 + i
        for ze in sig_res.events:
            if ze.index == abs_i and ze.kind in ("zone_entry", "zone_exit"):
                in_z = ze.kind == "zone_entry"
        entry_here = any((e.index - bar0) == i for e in ent_res.entries)
        sig_here   = any(abs(df.index - s.timestamp).argmin() == i
                         for s in sig_res.signals)
        col = ENTRY_COL if entry_here else (SIGNAL_COL if sig_here else
              (ZONE_BULL if in_z else "#21262d"))
        ax_bot.bar(i, 1, color=col, width=1, align="edge", linewidth=0)

    ax_bot.set_xlim(0, len(df))
    ax_bot.set_ylim(0, 1)
    ax_bot.set_yticks([])
    ax_bot.set_ylabel("Status", color="#8b949e", fontsize=8)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-1, len(df) + 14)
    step  = max(1, len(df) // 18)
    tpos  = list(range(0, len(df), step))
    tlbl  = [str(df.index[i])[:13] for i in tpos]
    for axis in (ax, ax_bot):
        axis.set_xticks(tpos)
    ax.set_xticklabels([""] * len(tpos))
    ax_bot.set_xticklabels(tlbl, rotation=35, ha="right",
                            color="#8b949e", fontsize=8)
    ax.yaxis.set_tick_params(colors="#8b949e")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Price (USD)", color="#8b949e")
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for spine in list(ax.spines.values()) + list(ax_bot.spines.values()):
        spine.set_color("#30363d")

    # ── Title ─────────────────────────────────────────────────────────────────
    vis = [e for e in ent_res.entries if 0 <= (e.index - bar0) < len(df)]
    ax.set_title(
        f"BTC-USD (5M)  —  MTF Entry Confirmation  "
        f"[1H bias={fib.direction.upper()}  zone=0.382-0.786  "
        f"timeout={ent_res.timeout_bars} bars]  "
        f"[{df.index[0].date()} - {df.index[-1].date()}]  |  "
        f"Entries total: {len(ent_res.entries)}  in view: {len(vis)}",
        color="#e6edf3", fontsize=10, pad=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=zone_col, alpha=0.35, label="Golden zone 38.2-61.8%"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=ENTRY_COL,
               markersize=14, lw=0, label="Entry (5M struct)"),
        Line2D([0],[0], color=SL_COL,  lw=1.5, label="Stop loss"),
        Line2D([0],[0], color=TP1_COL, lw=1.5, label="TP1 127.2%"),
        Line2D([0],[0], color=TP2_COL, lw=1.5, ls="--", label="TP2 161.8%"),
        Line2D([0],[0], color=SIGNAL_COL, lw=1, ls="--", label="15M signal"),
        mpatches.Patch(facecolor=BULL_CANDLE, label="Bull candle"),
        mpatches.Patch(facecolor=BEAR_CANDLE, label="Bear candle"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", ncol=4)

    plt.tight_layout(pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Chart saved -> {out_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Zoom chart  (single entry: 60 bars around it)
# ─────────────────────────────────────────────────────────────────────────────

def _zoom_chart(df_5m: pd.DataFrame, entry: EntrySignal,
                fib: FibResult, entry_n: int, out_path: Path) -> None:

    # 30 bars before entry, 60 after
    lo = max(0, entry.index - 30)
    hi = min(len(df_5m), entry.index + 60)
    df   = df_5m.iloc[lo:hi].copy()
    xi   = entry.index - lo   # entry x position in window

    matplotlib.rcParams.update({"font.family": "monospace", "font.size": 9})
    fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG)
    ax.set_facecolor(BG)

    _draw_candles(ax, df)

    # Fib zones
    zone_col = ZONE_BULL if fib.direction == "bullish" else ZONE_BEAR
    glo, ghi = fib.zone(0.382, 0.618)
    dlo, dhi = fib.zone(0.618, 0.786)
    ax.axhspan(glo, ghi, color=zone_col, alpha=0.25, zorder=1)
    ax.axhspan(dlo, dhi, color=zone_col, alpha=0.13, zorder=1)

    # All fib levels
    for lv in fib.levels:
        col = FIB_COLORS.get(lv.ratio, "#888")
        lw  = 1.3 if lv.kind == "extension" else 0.8
        ax.axhline(lv.price, color=col, lw=lw,
                   ls="--" if lv.kind == "extension" else "-",
                   alpha=0.8, zorder=4)
        ax.annotate(f" {lv.ratio*100:.1f}%  ${lv.price:,.0f}",
                    xy=(len(df)-1, lv.price), fontsize=7.5, color=col,
                    va="center", ha="left", zorder=7, annotation_clip=False)

    # Entry star
    ax.scatter(xi, entry.entry_price, color=ENTRY_COL, marker="*",
               s=350, zorder=10, edgecolors="#000", linewidths=0.5)

    # SL / TP full-width lines
    ax.axhline(entry.stop_loss, color=SL_COL,  lw=1.5, ls="-",  zorder=6, alpha=0.9)
    ax.axhline(entry.tp1,       color=TP1_COL, lw=1.5, ls="-",  zorder=6, alpha=0.9)
    ax.axhline(entry.tp2,       color=TP2_COL, lw=1.2, ls="--", zorder=6, alpha=0.9)

    # Shaded R:R box
    risk_pct   = abs(entry.entry_price - entry.stop_loss) / entry.entry_price * 100
    reward_pct = abs(entry.tp1 - entry.entry_price)       / entry.entry_price * 100
    ax.axhspan(entry.stop_loss, entry.entry_price,
               color=SL_COL,  alpha=0.08, zorder=0)
    ax.axhspan(entry.entry_price, entry.tp1,
               color=TP1_COL, alpha=0.08, zorder=0)

    # Annotations
    ax.annotate(f"  ENTRY #{entry_n}  ${entry.entry_price:,.2f}",
                xy=(xi, entry.entry_price), xytext=(xi+1, entry.entry_price),
                fontsize=9, color=ENTRY_COL, va="center", zorder=11)
    ax.annotate(f"  SL  ${entry.stop_loss:,.2f}  (-{risk_pct:.2f}%)",
                xy=(xi, entry.stop_loss), xytext=(xi+1, entry.stop_loss),
                fontsize=8, color=SL_COL, va="center", zorder=11)
    ax.annotate(f"  TP1  ${entry.tp1:,.2f}  (+{reward_pct:.2f}%)",
                xy=(xi, entry.tp1), xytext=(xi+1, entry.tp1),
                fontsize=8, color=TP1_COL, va="center", zorder=11)
    ax.annotate(f"  TP2  ${entry.tp2:,.2f}",
                xy=(xi, entry.tp2), xytext=(xi+1, entry.tp2),
                fontsize=8, color=TP2_COL, va="center", zorder=11)

    # Axes
    ax.set_xlim(-1, len(df) + 14)
    step  = max(1, len(df) // 12)
    tpos  = list(range(0, len(df), step))
    ax.set_xticks(tpos)
    ax.set_xticklabels([str(df.index[i])[:13] for i in tpos],
                       rotation=35, ha="right", color="#8b949e", fontsize=8)
    ax.yaxis.set_tick_params(colors="#8b949e")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    s = entry.confirmed_by
    ax.set_title(
        f"BTC-USD (5M)  —  Entry #{entry_n} zoom  |  "
        f"{entry.direction.upper()}  "
        f"confirmed by 5M {s.kind.upper()} {s.direction}  "
        f"|  R:R to TP1 = {entry.rr1:.2f}  "
        f"|  {str(entry.timestamp)[:16]}",
        color="#e6edf3", fontsize=10, pad=10,
    )

    handles = [
        Line2D([0],[0], marker="*", color="w", markerfacecolor=ENTRY_COL,
               markersize=12, lw=0, label=f"Entry  ${entry.entry_price:,.0f}"),
        Line2D([0],[0], color=SL_COL,  lw=1.5, label=f"SL  ${entry.stop_loss:,.0f}"),
        Line2D([0],[0], color=TP1_COL, lw=1.5, label=f"TP1  ${entry.tp1:,.0f}"),
        Line2D([0],[0], color=TP2_COL, lw=1.5, ls="--", label=f"TP2  ${entry.tp2:,.0f}"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

    plt.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Zoom chart saved -> {out_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="MTF entry test on BTC-USD")
    parser.add_argument("--n-bias",    type=int, default=7, choices=valid_n_values())
    parser.add_argument("--n-signal",  type=int, default=5, choices=valid_n_values())
    parser.add_argument("--n-entry",   type=int, default=3, choices=valid_n_values())
    parser.add_argument("--timeout",   type=int, default=4,
                        help="5M bars entry window stays open (default 4 = 20 min)")
    parser.add_argument("--zone-lo",   type=float, default=0.382)
    parser.add_argument("--zone-hi",   type=float, default=0.786)
    parser.add_argument("--bars",      type=int, default=300,
                        help="5M bars to show in overview chart")
    parser.add_argument("--zoom-entry", type=int, default=None,
                        help="Also generate a zoom chart for entry N (1-based)")
    args = parser.parse_args()

    console.print()
    console.print("[bold white]Hermes Trading — Step 5: 5M Entry Confirmation[/]")
    console.print(
        f"[dim]BIAS 1H N={args.n_bias}  |  SIGNAL 15M N={args.n_signal}  |  "
        f"ENTRY 5M N={args.n_entry}  |  timeout={args.timeout} bars[/]"
    )
    console.print()

    # ── Layer 1: 1H bias + fib ─────────────────────────────────────────────
    df_1h  = _fetch("BTC-USD", "1h",  "730d")
    sw_1h  = find_swings(df_1h,  n=args.n_bias)
    st_1h  = find_structure(df_1h,  sw_1h)
    fib    = fibs_from_last_choch(st_1h, sw_1h)
    bias   = st_1h.bias()

    if fib is None:
        console.print("[red]No CHoCH on 1H — cannot continue.[/]")
        return

    console.print(
        Columns([
            _kpi("1H bias",   bias.upper(), "green" if bias=="bullish" else "red"),
            _kpi("Fib dir",   fib.direction.upper()),
            _kpi("High",      f"${fib.high:,.2f}"),
            _kpi("Low",       f"${fib.low:,.2f}"),
            _kpi("Range",     f"{fib.range_pct:.2f}%"),
            _kpi("TP1",       f"${fib.tp1.price:,.2f}" if fib.tp1 else "-", "cyan"),
            _kpi("TP2",       f"${fib.tp2.price:,.2f}" if fib.tp2 else "-", "bold cyan"),
        ], equal=True)
    )
    console.print()

    # ── Layer 2: 15M signals ───────────────────────────────────────────────
    df_15m = _fetch("BTC-USD", "15m", "60d")
    sw_15m = find_swings(df_15m, n=args.n_signal)
    st_15m = find_structure(df_15m, sw_15m)
    sig    = find_signals(
        df_15m, fib=fib, bias=bias, struct_res=st_15m,
        zone_lo=args.zone_lo, zone_hi=args.zone_hi,
    )
    console.print(f"[dim]15M:[/]  {sig.summary()}")
    console.print()

    # ── Layer 3: 5M entries ────────────────────────────────────────────────
    df_5m  = _fetch("BTC-USD", "5m",  "60d")
    sw_5m  = find_swings(df_5m,  n=args.n_entry)
    st_5m  = find_structure(df_5m,  sw_5m)
    ent    = find_entries(
        df_5m, sig_res=sig, struct_res=st_5m, fib=fib,
        timeout_bars=args.timeout,
    )

    current = float(df_5m["close"].iloc[-1])

    console.print(
        Columns([
            _kpi("5M candles",   f"{len(df_5m):,}"),
            _kpi("5M swings",    str(len(sw_5m.points))),
            _kpi("5M struct",    str(len(st_5m.events))),
            _kpi("15M signals",  str(len(sig.signals))),
            _kpi("Entries",      str(len(ent.entries)), "bold cyan"),
            _kpi("Avg R:R1",     f"{ent.avg_rr1:.2f}",
                 "green" if ent.avg_rr1 >= 1.5 else "yellow"),
            _kpi("Current $",    f"${current:,.2f}"),
        ], equal=True)
    )
    console.print()

    if ent.entries:
        console.print(_entries_table(ent))
        console.print()

        last = ent.last_entry()
        console.print(f"[dim]Last entry:[/]  {last}")
        console.print()
    else:
        console.print("[yellow]No entries found in the 60-day 5M window.[/]")
        console.print("[dim]Try --no-struct on the signal step, or adjust --zone-lo / --zone-hi[/]")
        console.print()

    # ── Charts ────────────────────────────────────────────────────────────────
    out_ov = (Path(__file__).parent.parent / "charts"
              / f"entry_overview_1h15m5m_n{args.n_bias}{args.n_signal}{args.n_entry}.png")
    _overview_chart(df_5m, fib, sig, ent,
                    n_bars=min(args.bars, len(df_5m)), out_path=out_ov)

    if args.zoom_entry is not None:
        n = args.zoom_entry
        if 1 <= n <= len(ent.entries):
            entry_obj = ent.entries[n - 1]
            out_z = (Path(__file__).parent.parent / "charts"
                     / f"entry_zoom_{n}.png")
            _zoom_chart(df_5m, entry_obj, fib, n, out_z)
        else:
            console.print(f"[yellow]Entry #{n} not found (total {len(ent.entries)})[/]")

    console.print(f"[bold green]OK[/]  {ent.summary()}")
    console.print()


if __name__ == "__main__":
    main()
