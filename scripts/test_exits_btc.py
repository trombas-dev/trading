"""
test_exits_btc.py — Full pipeline: swing -> structure -> fib -> signal
                     -> entry -> exits  with performance report + charts.

Usage:
    uv run python scripts/test_exits_btc.py
    uv run python scripts/test_exits_btc.py --max-bars 150 --bars 400
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
from matplotlib.gridspec import GridSpec

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hermes_trading.strategy.swing import find_swings, valid_n_values
from hermes_trading.strategy.structure import find_structure
from hermes_trading.strategy.fibonacci import fibs_from_last_choch
from hermes_trading.strategy.signal import find_signals
from hermes_trading.strategy.entry import find_entries
from hermes_trading.strategy.exits import simulate_exits, TradeResult, ExitResult

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
BG          = "#0d1117"
GRID        = "#21262d"
BULL_CANDLE = "#26a641"
BEAR_CANDLE = "#da3633"
WIN_COL     = "#00e676"
LOSS_COL    = "#ff1744"
BE_COL      = "#ffa726"
TP1_COL     = "#69f0ae"
TP2_COL     = "#b9f6ca"
SL_COL      = "#f44336"
ENTRY_COL   = "#00e5ff"
ZONE_BULL   = "#1b5e20"
ZONE_BEAR   = "#b71c1c"

FIB_COLORS: dict[float, str] = {
    0.000: "#e6edf3", 0.236: "#ffe57f", 0.382: "#ffa726",
    0.500: "#ef6c00", 0.618: "#e53935", 0.786: "#880e4f",
    1.000: "#e6edf3", 1.272: "#00bcd4", 1.618: "#00e5ff",
}

STATUS_COLOR = {"win": WIN_COL, "loss": LOSS_COL,
                "breakeven": BE_COL, "partial": "#888", "open": "#888"}


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


def _trades_table(ex: ExitResult) -> Table:
    t = Table(
        title="Trade log — full pipeline",
        box=box.SIMPLE_HEAD, show_edge=False, title_style="bold",
    )
    t.add_column("#",        justify="right", style="dim")
    t.add_column("Entry ts", no_wrap=True)
    t.add_column("Entry $",  justify="right")
    t.add_column("SL $",     justify="right")
    t.add_column("TP1 $",    justify="right")
    t.add_column("Status",   justify="center")
    t.add_column("Exit ts",  no_wrap=True, style="dim")
    t.add_column("Exit $",   justify="right")
    t.add_column("Bars",     justify="right", style="dim")
    t.add_column("P&L R",    justify="right")
    t.add_column("Exits")

    for n, tr in enumerate(ex.trades, 1):
        st     = tr.status
        sc     = STATUS_COLOR.get(st, "white")
        fe     = tr.final_exit
        pnl_r  = tr.total_pnl_r
        pnl_st = "bold green" if pnl_r > 0.05 else ("red" if pnl_r < -0.05 else "dim")

        exits_str = " -> ".join(e.kind[:2].upper() for e in tr.exits)

        t.add_row(
            str(n),
            str(tr.entry.timestamp)[:16],
            f"[{ENTRY_COL}]${tr.entry.entry_price:,.2f}[/]",
            f"[red]${tr.entry.stop_loss:,.2f}[/]",
            f"[cyan]${tr.entry.tp1:,.2f}[/]",
            Text(st.upper(), style=f"bold {sc}"),
            str(fe.timestamp)[:16] if fe else "-",
            f"${fe.price:,.2f}"    if fe else "-",
            str(tr.duration_bars),
            Text(f"{pnl_r:+.3f}R", style=pnl_st),
            exits_str,
        )
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick renderer
# ─────────────────────────────────────────────────────────────────────────────

def _draw_candles(ax, df: pd.DataFrame) -> None:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    w = 0.6
    for i in range(len(df)):
        col = BULL_CANDLE if c[i] >= o[i] else BEAR_CANDLE
        ax.plot([i, i], [l[i], h[i]], color=col, lw=0.6, zorder=2)
        lo, hi = min(o[i], c[i]), max(o[i], c[i])
        if hi - lo < 1e-8:
            ax.plot([i-w/2, i+w/2], [c[i], c[i]], color=col, lw=1, zorder=3)
        else:
            ax.add_patch(mpatches.FancyBboxPatch(
                (i-w/2, lo), w, hi-lo,
                boxstyle="square,pad=0", facecolor=col, edgecolor="none", zorder=3))


# ─────────────────────────────────────────────────────────────────────────────
# Main chart: price + trades + equity curve
# ─────────────────────────────────────────────────────────────────────────────

def _save_chart(df_5m: pd.DataFrame, ex: ExitResult, fib,
                n_bars: int, out_path: Path) -> None:

    df   = df_5m.iloc[-n_bars:].copy()
    bar0 = len(df_5m) - n_bars

    matplotlib.rcParams.update({"font.family": "monospace", "font.size": 9})

    fig = plt.figure(figsize=(24, 14), facecolor=BG)
    gs  = GridSpec(3, 1, figure=fig,
                   height_ratios=[5, 1.2, 1.2], hspace=0.08)
    ax_price  = fig.add_subplot(gs[0])
    ax_equity = fig.add_subplot(gs[1])
    ax_bot    = fig.add_subplot(gs[2])
    for ax in (ax_price, ax_equity, ax_bot):
        ax.set_facecolor(BG)

    # ── Candlesticks ──────────────────────────────────────────────────────────
    _draw_candles(ax_price, df)

    # ── Fib zones ─────────────────────────────────────────────────────────────
    zone_col = ZONE_BULL if fib.direction == "bullish" else ZONE_BEAR
    glo, ghi = fib.zone(0.382, 0.618)
    dlo, dhi = fib.zone(0.618, 0.786)
    ax_price.axhspan(glo, ghi, color=zone_col, alpha=0.20, zorder=1)
    ax_price.axhspan(dlo, dhi, color=zone_col, alpha=0.10, zorder=1)

    # ── Fib level lines (right-side labels) ───────────────────────────────────
    for lv in fib.levels:
        col = FIB_COLORS.get(lv.ratio, "#888")
        lw  = 1.2 if lv.kind == "extension" else 0.6
        ax_price.axhline(lv.price, color=col, lw=lw,
                         ls="--" if lv.kind == "extension" else "-",
                         alpha=0.60, zorder=4)
        ax_price.annotate(
            f" {lv.ratio*100:.1f}%  ${lv.price:,.0f}",
            xy=(len(df)-1, lv.price), fontsize=7, color=col,
            va="center", ha="left", zorder=7, annotation_clip=False)

    # ── Trades ────────────────────────────────────────────────────────────────
    for n_trade, tr in enumerate(ex.trades, 1):
        xi_entry = tr.entry.index - bar0
        if xi_entry < -10 or xi_entry >= len(df) + 10:
            continue

        fe     = tr.final_exit
        xi_end = (fe.index - bar0) if fe else xi_entry + 20
        st     = tr.status
        col    = WIN_COL if st == "win" else (LOSS_COL if st == "loss" else BE_COL)

        # Shaded trade region
        if 0 <= xi_entry < len(df):
            x_lo = max(0, xi_entry)
            x_hi = min(len(df), xi_end)
            ep   = tr.entry.entry_price
            sl   = tr.entry.stop_loss
            tp1  = tr.entry.tp1
            shade_hi = max(ep, tp1)
            shade_lo = min(ep, sl)
            ax_price.add_patch(mpatches.FancyBboxPatch(
                (x_lo, shade_lo), x_hi - x_lo, shade_hi - shade_lo,
                boxstyle="square,pad=0", facecolor=col,
                alpha=0.07, zorder=1, linewidth=0))

        # Entry star
        if 0 <= xi_entry < len(df):
            ax_price.scatter(xi_entry, tr.entry.entry_price,
                             color=ENTRY_COL, marker="*", s=200, zorder=9,
                             edgecolors="#000", linewidths=0.4)

        # Exit marker
        if fe and 0 <= xi_end < len(df):
            mrkr = "o" if st == "win" else ("x" if st == "loss" else "s")
            ax_price.scatter(xi_end, fe.price,
                             color=col, marker=mrkr, s=80, zorder=9,
                             edgecolors="#000", linewidths=0.4)

        # SL / TP snippets from entry
        if 0 <= xi_entry < len(df):
            xe = min(xi_entry + 30, len(df) - 1)
            ax_price.hlines(tr.entry.stop_loss, xi_entry, xe,
                            colors=SL_COL,  lw=0.9, ls="-",  zorder=5, alpha=0.7)
            ax_price.hlines(tr.entry.tp1,     xi_entry, xe,
                            colors=TP1_COL, lw=0.9, ls="-",  zorder=5, alpha=0.7)
            ax_price.hlines(tr.entry.tp2,     xi_entry, xe,
                            colors=TP2_COL, lw=0.8, ls="--", zorder=5, alpha=0.5)

        # Trade label
        if 0 <= xi_entry < len(df):
            pnl = tr.total_pnl_r
            ax_price.annotate(
                f"#{n_trade}\n{pnl:+.1f}R",
                xy=(xi_entry, tr.entry.entry_price),
                xytext=(xi_entry + 1,
                        tr.entry.entry_price * (1.002 if fib.direction=="bullish"
                                                else 0.998)),
                fontsize=7, color=col, zorder=10, annotation_clip=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    eq = ex.equity_curve
    if eq:
        xs_eq = list(range(len(eq)))
        ax_equity.plot(xs_eq, eq, color=WIN_COL, lw=1.5, zorder=3)
        ax_equity.fill_between(xs_eq, 0, eq,
                               where=[v >= 0 for v in eq],
                               color=WIN_COL, alpha=0.15)
        ax_equity.fill_between(xs_eq, 0, eq,
                               where=[v < 0 for v in eq],
                               color=LOSS_COL, alpha=0.15)
        ax_equity.axhline(0, color="#555", lw=0.8, zorder=2)
        ax_equity.set_ylabel("Cum. R", color="#8b949e", fontsize=8)
        ax_equity.yaxis.set_tick_params(colors="#8b949e")
        ax_equity.set_xlim(-0.5, max(len(eq) - 0.5, 0.5))
        # R grid lines
        for r in [-1, 0, 1, 2, 3]:
            ax_equity.axhline(r, color=GRID, lw=0.4)

    # ── Per-trade R bar chart ─────────────────────────────────────────────────
    for i, tr in enumerate(ex.closed):
        r   = tr.total_pnl_r
        col = WIN_COL if r > 0.05 else (LOSS_COL if r < -0.05 else BE_COL)
        ax_bot.bar(i, r, color=col, width=0.7, zorder=3)
    ax_bot.axhline(0, color="#555", lw=0.8, zorder=2)
    ax_bot.set_ylabel("R / trade", color="#8b949e", fontsize=8)
    ax_bot.yaxis.set_tick_params(colors="#8b949e")
    if ex.closed:
        ax_bot.set_xlim(-0.5, len(ex.closed) - 0.5)

    # ── X-axis ────────────────────────────────────────────────────────────────
    step  = max(1, len(df) // 18)
    tpos  = list(range(0, len(df), step))
    tlbl  = [str(df.index[i])[:13] for i in tpos]
    ax_price.set_xticks(tpos)
    ax_price.set_xticklabels([""] * len(tpos))
    ax_price.set_xlim(-1, len(df) + 14)
    ax_price.yaxis.set_tick_params(colors="#8b949e")
    ax_price.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_price.set_ylabel("Price (USD)", color="#8b949e")
    ax_price.grid(True, color=GRID, lw=0.4, zorder=0)
    for ax in (ax_price, ax_equity, ax_bot):
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax_price.set_title(
        f"BTC-USD (5M)  —  Full MTF Pipeline: Swing -> Structure -> Fib -> "
        f"Signal -> Entry -> Exits  |  "
        f"Bias={fib.direction.upper()}  "
        f"Trades={len(ex.trades)}  Closed={len(ex.closed)}  "
        f"WR={ex.win_rate:.0f}%  "
        f"TotalR={ex.total_pnl_r:+.2f}  "
        f"PF={ex.profit_factor:.2f}",
        color="#e6edf3", fontsize=10, pad=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        Line2D([0],[0], marker="*", color="w", markerfacecolor=ENTRY_COL,
               markersize=11, lw=0, label="Entry"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=WIN_COL,
               markersize=8,  lw=0, label="Win exit"),
        Line2D([0],[0], marker="x", color=LOSS_COL, markersize=8, lw=0,
               markeredgewidth=2, label="Stop loss"),
        Line2D([0],[0], color=SL_COL,  lw=1.5, label="SL"),
        Line2D([0],[0], color=TP1_COL, lw=1.5, label="TP1 127.2%"),
        Line2D([0],[0], color=TP2_COL, lw=1.5, ls="--", label="TP2 161.8%"),
        mpatches.Patch(facecolor=zone_col, alpha=0.30, label="Golden zone"),
    ]
    ax_price.legend(handles=handles, loc="upper left", fontsize=8,
                    facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#e6edf3", ncol=4)

    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Chart saved -> {out_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Full MTF exits test on BTC-USD")
    parser.add_argument("--n-bias",   type=int, default=7, choices=valid_n_values())
    parser.add_argument("--n-signal", type=int, default=5, choices=valid_n_values())
    parser.add_argument("--n-entry",  type=int, default=3, choices=valid_n_values())
    parser.add_argument("--zone-lo",  type=float, default=0.382)
    parser.add_argument("--zone-hi",  type=float, default=0.786)
    parser.add_argument("--timeout",  type=int, default=4)
    parser.add_argument("--max-bars", type=int, default=100,
                        help="Max 5M bars a trade stays open (default 100 = ~8h)")
    parser.add_argument("--bars",     type=int, default=400,
                        help="5M bars to display in chart (default 400)")
    args = parser.parse_args()

    console.print()
    console.print("[bold white]Hermes Trading — Step 6: Exits & Performance[/]")
    console.print(
        f"[dim]BIAS 1H N={args.n_bias}  |  SIGNAL 15M N={args.n_signal}  |  "
        f"ENTRY 5M N={args.n_entry}  |  max_bars={args.max_bars}[/]"
    )
    console.print()

    # ── Full pipeline ──────────────────────────────────────────────────────────
    df_1h  = _fetch("BTC-USD", "1h",  "730d")
    sw_1h  = find_swings(df_1h,  n=args.n_bias)
    st_1h  = find_structure(df_1h,  sw_1h)
    fib    = fibs_from_last_choch(st_1h, sw_1h)
    bias   = st_1h.bias()
    if fib is None:
        console.print("[red]No 1H CHoCH found.[/]"); return

    df_15m = _fetch("BTC-USD", "15m", "60d")
    sw_15m = find_swings(df_15m, n=args.n_signal)
    st_15m = find_structure(df_15m, sw_15m)
    sig    = find_signals(df_15m, fib=fib, bias=bias, struct_res=st_15m,
                          zone_lo=args.zone_lo, zone_hi=args.zone_hi)

    df_5m  = _fetch("BTC-USD", "5m",  "60d")
    sw_5m  = find_swings(df_5m,  n=args.n_entry)
    st_5m  = find_structure(df_5m,  sw_5m)
    ent    = find_entries(df_5m, sig_res=sig, struct_res=st_5m, fib=fib,
                          timeout_bars=args.timeout)
    ex     = simulate_exits(df_5m, ent, max_bars=args.max_bars)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    console.print(
        Columns([
            _kpi("1H bias",       bias.upper(),
                 "green" if bias=="bullish" else "red"),
            _kpi("Trades",        str(len(ex.trades))),
            _kpi("Closed",        str(len(ex.closed))),
            _kpi("Wins",          str(len(ex.wins)),      "green"),
            _kpi("Losses",        str(len(ex.losses)),    "red"),
            _kpi("Win rate",      f"{ex.win_rate:.1f}%",
                 "green" if ex.win_rate >= 50 else "yellow"),
            _kpi("Total R",       f"{ex.total_pnl_r:+.2f}R",
                 "green" if ex.total_pnl_r > 0 else "red"),
            _kpi("Avg R / trade", f"{ex.avg_pnl_r:+.2f}R",
                 "green" if ex.avg_pnl_r > 0 else "red"),
        ], equal=True)
    )
    console.print(
        Columns([
            _kpi("Profit factor", f"{ex.profit_factor:.2f}",
                 "green" if ex.profit_factor >= 1.5 else "yellow"),
            _kpi("Best trade",    f"{ex.max_win_r:+.2f}R",  "green"),
            _kpi("Worst trade",   f"{ex.max_loss_r:+.2f}R", "red"),
            _kpi("Breakevens",    str(len(ex.breakevens))),
            _kpi("TP1 range",     f"${fib.tp1.price:,.0f}" if fib.tp1 else "-", "cyan"),
            _kpi("TP2 range",     f"${fib.tp2.price:,.0f}" if fib.tp2 else "-", "bold cyan"),
            _kpi("Fib range",     f"{fib.range_pct:.2f}%"),
            _kpi("Max bars/trade",str(args.max_bars)),
        ], equal=True)
    )
    console.print()

    # ── Exit breakdown ────────────────────────────────────────────────────────
    exit_counts: dict[str, int] = {}
    for tr in ex.closed:
        for ev in tr.exits:
            exit_counts[ev.kind] = exit_counts.get(ev.kind, 0) + 1

    breakdown = Table(title="Exit breakdown", box=box.SIMPLE_HEAD,
                      show_edge=False, title_style="dim")
    breakdown.add_column("Exit type"); breakdown.add_column("Count", justify="right")
    breakdown.add_column("% of exit events", justify="right")
    total_evts = sum(exit_counts.values())
    for kind, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_evts * 100 if total_evts else 0
        col = "green" if kind in ("tp1","tp2") else ("red" if kind=="stop_loss" else "dim")
        breakdown.add_row(Text(kind, style=col), str(cnt), f"{pct:.1f}%")
    console.print(breakdown)
    console.print()

    # ── Trade log ─────────────────────────────────────────────────────────────
    console.print(_trades_table(ex))
    console.print()

    # ── Equity curve in text ──────────────────────────────────────────────────
    eq = ex.equity_curve
    if eq:
        console.print(
            f"[dim]Equity curve:[/]  "
            f"start=0.00R  "
            f"peak=[green]{max(eq):+.2f}R[/]  "
            f"trough=[red]{min(eq):+.2f}R[/]  "
            f"final=[{'green' if eq[-1]>0 else 'red'}]{eq[-1]:+.2f}R[/]"
        )
        console.print()

    # ── Chart ─────────────────────────────────────────────────────────────────
    out = (Path(__file__).parent.parent / "charts" / "exits_full_pipeline.png")
    _save_chart(df_5m, ex, fib, n_bars=min(args.bars, len(df_5m)), out_path=out)

    console.print(f"[bold green]OK[/]  {ex.summary()}")
    console.print()


if __name__ == "__main__":
    main()
