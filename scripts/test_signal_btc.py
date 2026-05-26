"""
test_signal_btc.py — Multi-timeframe signal detection test on BTC-USD.

Pipeline:
  1H  -> swings -> structure -> fib (bias)
  15M -> swings -> structure -> signals  (zone + 15M confirmation)

Usage:
    uv run python scripts/test_signal_btc.py
    uv run python scripts/test_signal_btc.py --n-bias 7 --n-signal 5 --bars 300
    uv run python scripts/test_signal_btc.py --zone-lo 0.382 --zone-hi 0.618
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
from hermes_trading.strategy.signal import find_signals, SignalEvent, SignalResult

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Colours (consistent across all scripts)
# ─────────────────────────────────────────────────────────────────────────────
BG           = "#0d1117"
GRID         = "#21262d"
BULL_CANDLE  = "#26a641"
BEAR_CANDLE  = "#da3633"

FIB_COLORS: dict[float, str] = {
    0.000: "#e6edf3",
    0.236: "#ffe57f",
    0.382: "#ffa726",
    0.500: "#ef6c00",
    0.618: "#e53935",
    0.786: "#880e4f",
    1.000: "#e6edf3",
    1.272: "#00bcd4",
    1.618: "#00e5ff",
}

SIGNAL_COLOR     = "#ffeb3b"   # yellow star — confirmed signal
ZONE_ENTRY_COLOR = "#66bb6a"   # light green — price enters zone
ZONE_EXIT_COLOR  = "#ef5350"   # light red   — price leaves zone


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(ticker: str, interval: str, period: str) -> pd.DataFrame:
    console.print(f"[dim]Downloading {ticker}  interval={interval}  period={period} ...[/]")
    raw = yf.download(ticker, interval=interval, period=period,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"No data for {ticker} {interval}")
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


def _signals_table(sig_res: SignalResult, n_rows: int = 30) -> Table:
    t = Table(
        title=f"Signal Events  (bias={sig_res.bias}  zone={sig_res.zone_lo:.3f}-{sig_res.zone_hi:.3f})",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        title_style="bold",
    )
    t.add_column("Bar #",     justify="right", style="dim")
    t.add_column("Timestamp", no_wrap=True)
    t.add_column("Kind",      justify="center")
    t.add_column("Price",     justify="right")
    t.add_column("Nearest fib",  style="dim")
    t.add_column("Zone",      justify="center")
    t.add_column("15M struct")

    for e in sig_res.events[-n_rows:]:
        if e.kind == "signal":
            kind_text = Text("SIGNAL", style="bold yellow")
        elif e.kind == "zone_entry":
            kind_text = Text("ENTER", style="green")
        else:
            kind_text = Text("EXIT",  style="red")

        zone_str   = "GOLDEN" if e.in_golden else ("DEEP" if e.in_deep else "-")
        zone_style = "green"  if e.in_golden else ("yellow" if e.in_deep else "dim")

        struct_str = ""
        if e.struct_event:
            s = e.struct_event
            sc = "green" if s.direction == "bullish" else "red"
            struct_str = f"[{sc}]{s.kind.upper()} {s.direction}[/]"

        t.add_row(
            str(e.index),
            str(e.timestamp)[:16],
            kind_text,
            f"${e.price:,.2f}",
            e.nearest_fib.label.strip(),
            Text(zone_str, style=zone_style),
            struct_str,
        )
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick renderer
# ─────────────────────────────────────────────────────────────────────────────

def _draw_candles(ax, df: pd.DataFrame) -> None:
    opens  = df["open"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    xs = np.arange(len(df))
    w  = 0.6
    for i in range(len(df)):
        col = BULL_CANDLE if closes[i] >= opens[i] else BEAR_CANDLE
        ax.plot([xs[i], xs[i]], [lows[i], highs[i]], color=col, lw=0.7, zorder=2)
        lo = min(opens[i], closes[i])
        hi = max(opens[i], closes[i])
        if hi - lo < 1e-8:
            ax.plot([xs[i]-w/2, xs[i]+w/2], [closes[i], closes[i]],
                    color=col, lw=1, zorder=3)
        else:
            ax.add_patch(mpatches.FancyBboxPatch(
                (xs[i]-w/2, lo), w, hi-lo,
                boxstyle="square,pad=0", facecolor=col, edgecolor="none", zorder=3))


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def _save_chart(df_15m: pd.DataFrame, fib: FibResult,
                sig_res: SignalResult, n_signal: int,
                n_bars: int, out_path: Path) -> None:

    df   = df_15m.iloc[-n_bars:].copy()
    bar0 = len(df_15m) - n_bars    # absolute bar index of first visible bar

    matplotlib.rcParams.update({"font.family": "monospace", "font.size": 9})
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(22, 13), facecolor=BG,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.06},
    )
    ax.set_facecolor(BG)
    ax2.set_facecolor(BG)

    # ── Candlesticks ──────────────────────────────────────────────────────────
    _draw_candles(ax, df)

    # ── Fibonacci zone bands ──────────────────────────────────────────────────
    zone_col  = "#1b5e20" if fib.direction == "bullish" else "#b71c1c"
    golden_lo, golden_hi = fib.zone(0.382, 0.618)
    deep_lo,   deep_hi   = fib.zone(0.618, 0.786)
    ax.axhspan(golden_lo, golden_hi, color=zone_col, alpha=0.22, zorder=1)
    ax.axhspan(deep_lo,   deep_hi,   color=zone_col, alpha=0.12, zorder=1)

    # ── Fibonacci level lines ─────────────────────────────────────────────────
    for lv in fib.levels:
        col  = FIB_COLORS.get(lv.ratio, "#888")
        lw   = 1.2 if lv.kind == "extension" else 0.7
        ls   = "--" if lv.kind == "extension" else "-"
        ax.axhline(lv.price, color=col, lw=lw, ls=ls, alpha=0.7, zorder=4)
        ax.annotate(
            f" {lv.ratio*100:.1f}%  ${lv.price:,.0f}",
            xy=(len(df)-1, lv.price), fontsize=7, color=col,
            va="center", ha="left", zorder=7, annotation_clip=False,
        )

    # ── Signal events ─────────────────────────────────────────────────────────
    visible_events = [e for e in sig_res.events if e.index >= bar0]

    for e in visible_events:
        xi = e.index - bar0
        if xi < 0 or xi >= len(df):
            continue

        if e.kind == "signal":
            ax.scatter(xi, e.price, color=SIGNAL_COLOR, marker="*",
                       s=220, zorder=8, linewidths=0.5, edgecolors="#000")
            ax.annotate(
                f" SIGNAL\n ${e.price:,.0f}",
                xy=(xi, e.price),
                xytext=(xi + 1, e.price * (1.0015 if fib.direction == "bullish" else 0.9985)),
                fontsize=7.5, color=SIGNAL_COLOR, va="bottom", zorder=9,
                annotation_clip=True,
            )
        elif e.kind == "zone_entry":
            ax.axvline(xi, color=ZONE_ENTRY_COLOR, lw=0.8, ls=":", alpha=0.6, zorder=5)
        elif e.kind == "zone_exit":
            ax.axvline(xi, color=ZONE_EXIT_COLOR, lw=0.8, ls=":", alpha=0.6, zorder=5)

    # ── Lower panel: zone state bar ───────────────────────────────────────────
    # Paint each bar green (in zone) or grey (out of zone)
    in_z = False
    for i in range(len(df)):
        abs_i = bar0 + i
        zone_events_here = [e for e in sig_res.events
                            if e.index == abs_i
                            and e.kind in ("zone_entry", "zone_exit")]
        for ze in zone_events_here:
            in_z = (ze.kind == "zone_entry")
        sig_here = any(e.index == abs_i and e.kind == "signal"
                       for e in sig_res.events)
        color = SIGNAL_COLOR if sig_here else (
            ZONE_ENTRY_COLOR if in_z else "#21262d")
        ax2.bar(i, 1, color=color, width=1, align="edge", linewidth=0)

    ax2.set_xlim(0, len(df))
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_facecolor(BG)
    ax2.set_ylabel("In zone", color="#8b949e", fontsize=8)

    # ── X-axis (shared) ───────────────────────────────────────────────────────
    step = max(1, len(df) // 18)
    tpos = list(range(0, len(df), step))
    tlbl = [str(df.index[i])[:13] for i in tpos]
    for axis in (ax, ax2):
        axis.set_xticks(tpos)
    ax.set_xticklabels([""] * len(tpos))
    ax2.set_xticklabels(tlbl, rotation=35, ha="right", color="#8b949e", fontsize=8)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_xlim(-1, len(df) + 14)
    ax.set_ylabel("Price (USD)", color="#8b949e")
    ax.yaxis.set_tick_params(colors="#8b949e")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    for spine in ax2.spines.values():
        spine.set_color("#30363d")

    # ── Title ─────────────────────────────────────────────────────────────────
    n_sig = len([e for e in visible_events if e.kind == "signal"])
    ax.set_title(
        f"BTC-USD (15M)  —  MTF Signal Detection  "
        f"[1H bias={fib.direction.upper()}  N-signal={n_signal}  "
        f"zone={sig_res.zone_lo:.3f}-{sig_res.zone_hi:.3f}]  "
        f"[{df.index[0].date()} - {df.index[-1].date()}]  |  "
        f"Signals in view: {n_sig}",
        color="#e6edf3", fontsize=10, pad=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=zone_col, alpha=0.35, label="Golden zone 38.2-61.8%"),
        mpatches.Patch(facecolor=zone_col, alpha=0.18, label="Deep zone 61.8-78.6%"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=SIGNAL_COLOR,
               markersize=12, lw=0, label="Signal (zone + 15M struct)"),
        Line2D([0],[0], color=ZONE_ENTRY_COLOR, lw=1, ls=":", label="Zone entry"),
        Line2D([0],[0], color=ZONE_EXIT_COLOR,  lw=1, ls=":", label="Zone exit"),
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="MTF signal test on BTC-USD")
    parser.add_argument("--n-bias",   type=int, default=7, choices=valid_n_values(),
                        help="N for swing detection on 1H bias TF (default 7)")
    parser.add_argument("--n-signal", type=int, default=5, choices=valid_n_values(),
                        help="N for swing detection on 15M signal TF (default 5)")
    parser.add_argument("--zone-lo",  type=float, default=0.382,
                        help="Lower fib bound of entry zone (default 0.382)")
    parser.add_argument("--zone-hi",  type=float, default=0.786,
                        help="Upper fib bound of entry zone (default 0.786)")
    parser.add_argument("--bars",     type=int, default=200,
                        help="15M bars to show in chart (default 200)")
    parser.add_argument("--no-struct", action="store_true",
                        help="Signal on zone entry alone (skip 15M structure requirement)")
    args = parser.parse_args()

    console.print()
    console.print("[bold white]Hermes Trading — Step 4: MTF Signal Detection[/]")
    console.print(
        f"[dim]BIAS: 1H N={args.n_bias}   "
        f"SIGNAL: 15M N={args.n_signal}   "
        f"zone {args.zone_lo:.3f}-{args.zone_hi:.3f}   "
        f"require_struct={not args.no_struct}[/]"
    )
    console.print()

    # ── 1H bias layer ─────────────────────────────────────────────────────────
    df_1h = _fetch("BTC-USD", "1h", "730d")
    sw_1h = find_swings(df_1h, n=args.n_bias)
    st_1h = find_structure(df_1h, sw_1h)
    fib   = fibs_from_last_choch(st_1h, sw_1h)
    bias  = st_1h.bias()

    if fib is None:
        console.print("[red]No CHoCH found on 1H — cannot build Fibonacci levels.[/]")
        return

    console.print(
        Columns([
            _kpi("1H bias",        bias.upper(),
                 "green" if bias == "bullish" else "red"),
            _kpi("Fib direction",  fib.direction.upper()),
            _kpi("Swing High",     f"${fib.high:,.2f}"),
            _kpi("Swing Low",      f"${fib.low:,.2f}"),
            _kpi("Range",          f"{fib.range_pct:.2f}%"),
            _kpi("TP1 (127.2%)",   f"${fib.tp1.price:,.2f}" if fib.tp1 else "-", "cyan"),
            _kpi("TP2 (161.8%)",   f"${fib.tp2.price:,.2f}" if fib.tp2 else "-", "bold cyan"),
        ], equal=True)
    )
    console.print()

    # ── 15M signal layer ──────────────────────────────────────────────────────
    df_15m = _fetch("BTC-USD", "15m", "60d")
    sw_15m = find_swings(df_15m, n=args.n_signal)
    st_15m = find_structure(df_15m, sw_15m)
    sig    = find_signals(
        df_15m, fib=fib, bias=bias, struct_res=st_15m,
        zone_lo=args.zone_lo, zone_hi=args.zone_hi,
        require_struct=not args.no_struct,
    )

    current_price = float(df_15m["close"].iloc[-1])
    in_zone_now   = sig.currently_in_zone(len(df_15m) - 1)
    in_golden     = fib.is_in_zone(current_price, 0.382, 0.618)
    in_deep       = fib.is_in_zone(current_price, 0.618, 0.786)
    zone_txt      = "GOLDEN ZONE" if in_golden else ("DEEP ZONE" if in_deep else "outside zone")
    zone_style    = "bold green" if in_golden else ("bold yellow" if in_deep else "dim")

    console.print(
        Columns([
            _kpi("15M candles",    f"{len(df_15m):,}"),
            _kpi("15M swings",     str(len(sw_15m.points))),
            _kpi("15M struct",     str(len(st_15m.events))),
            _kpi("Zone entries",   str(len(sig.zone_entries))),
            _kpi("Zone exits",     str(len(sig.zone_exits))),
            _kpi("Signals",        str(len(sig.signals)), "bold yellow"),
            _kpi("Current price",  f"${current_price:,.2f}"),
            _kpi("Zone status",    zone_txt, zone_style),
        ], equal=True)
    )
    console.print()

    # ── Last signal ───────────────────────────────────────────────────────────
    last_sig = sig.last_signal()
    if last_sig:
        console.print(f"[dim]Last signal:[/]  {last_sig}")
        console.print()

    # ── Events table ──────────────────────────────────────────────────────────
    console.print(_signals_table(sig, n_rows=30))
    console.print()

    # ── Signals only ──────────────────────────────────────────────────────────
    if sig.signals:
        t2 = Table(
            title=f"Confirmed signals only  ({len(sig.signals)} total)",
            box=box.SIMPLE_HEAD, show_edge=False, title_style="bold yellow",
        )
        t2.add_column("Bar #",     justify="right", style="dim")
        t2.add_column("Timestamp", no_wrap=True)
        t2.add_column("Price",     justify="right")
        t2.add_column("Zone",      justify="center")
        t2.add_column("15M struct confirm")
        for e in sig.signals[-20:]:
            z  = Text("GOLDEN", "green") if e.in_golden else Text("DEEP", "yellow")
            sc = ""
            if e.struct_event:
                s  = e.struct_event
                c  = "green" if s.direction == "bullish" else "red"
                sc = f"[{c}]{s.kind.upper()} {s.direction}  level=${s.level:,.2f}[/]"
            t2.add_row(str(e.index), str(e.timestamp)[:16],
                       f"${e.price:,.2f}", z, sc)
        console.print(t2)
        console.print()

    # ── Chart ─────────────────────────────────────────────────────────────────
    out = (Path(__file__).parent.parent / "charts"
           / f"signal_1h_15m_n{args.n_bias}_{args.n_signal}.png")
    _save_chart(
        df_15m, fib, sig,
        n_signal=args.n_signal,
        n_bars=min(args.bars, len(df_15m)),
        out_path=out,
    )

    console.print(f"[bold green]OK[/]  {sig.summary()}")
    console.print()


if __name__ == "__main__":
    main()
