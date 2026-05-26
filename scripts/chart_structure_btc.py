"""
chart_structure_btc.py — High-quality BOS / CHoCH price chart.

Draws a candlestick-style chart for BTC-USD with every BOS and CHoCH
event annotated with its broken price level.

Usage:
    uv run python scripts/chart_structure_btc.py           # last 365 daily bars
    uv run python scripts/chart_structure_btc.py --bars 600
    uv run python scripts/chart_structure_btc.py --tf 1h --bars 300 --n 5
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
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from hermes_trading.strategy.swing import find_swings, valid_n_values
from hermes_trading.strategy.structure import find_structure, StructureEvent

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG         = "#0d1117"
GRID       = "#21262d"
BULL_CANDLE = "#26a641"
BEAR_CANDLE = "#da3633"
BOS_BULL   = "#3fb950"   # green
BOS_BEAR   = "#f85149"   # red
CHOCH_BULL = "#f0883e"   # orange
CHOCH_BEAR = "#a371f7"   # purple
LEVEL_ALPHA = 0.35


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
    print(f"Downloading {ticker}  interval={yf_interval}  period={yf_period} ...")
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
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick renderer
# ─────────────────────────────────────────────────────────────────────────────

def _draw_candles(ax, df: pd.DataFrame) -> None:
    """Draw OHLC candlesticks using numeric x-axis positions."""
    xs = np.arange(len(df))
    opens  = df["open"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    width  = 0.6

    for i in range(len(df)):
        bull = closes[i] >= opens[i]
        color = BULL_CANDLE if bull else BEAR_CANDLE

        # High-low wick
        ax.plot([xs[i], xs[i]], [lows[i], highs[i]],
                color=color, linewidth=0.7, zorder=2)

        # Body
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        if body_hi - body_lo < 1e-8:          # doji — draw a thin line
            ax.plot([xs[i] - width / 2, xs[i] + width / 2],
                    [closes[i], closes[i]], color=color, linewidth=1, zorder=3)
        else:
            rect = mpatches.FancyBboxPatch(
                (xs[i] - width / 2, body_lo),
                width,
                body_hi - body_lo,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
            ax.add_patch(rect)


# ─────────────────────────────────────────────────────────────────────────────
# Main chart
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(df_full: pd.DataFrame, n: int, n_bars: int, tf: str,
                out_path: Path) -> None:

    # ── Run detectors on the FULL history (no lookahead) ─────────────────────
    swing_res  = find_swings(df_full, n=n)
    struct_res = find_structure(df_full, swing_res)

    # ── Slice the view window ─────────────────────────────────────────────────
    df = df_full.iloc[-n_bars:].copy()
    bar_offset = len(df_full) - n_bars   # absolute index of df.iloc[0]

    # Events whose close bar falls inside the view window
    visible_events: list[StructureEvent] = [
        e for e in struct_res.events
        if e.index >= bar_offset
    ]

    # ── Figure setup ─────────────────────────────────────────────────────────
    matplotlib.rcParams.update({
        "font.family": "monospace",
        "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(22, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # ── Candlesticks ──────────────────────────────────────────────────────────
    _draw_candles(ax, df)

    # ── Structure events ──────────────────────────────────────────────────────
    label_budget: dict[str, float] = {}   # price → last x where we placed a label

    for e in visible_events:
        xi = e.index - bar_offset          # x position in view
        if xi < 0 or xi >= len(df):
            continue

        # Colour / marker / label prefix
        if e.kind == "bos" and e.direction == "bullish":
            color, marker, prefix = BOS_BULL,   "^", "BOS▲"
        elif e.kind == "bos" and e.direction == "bearish":
            color, marker, prefix = BOS_BEAR,   "v", "BOS▼"
        elif e.kind == "choch" and e.direction == "bullish":
            color, marker, prefix = CHOCH_BULL, "D", "CHoCH▲"
        else:
            color, marker, prefix = CHOCH_BEAR, "D", "CHoCH▼"

        y_event = e.close

        # ── Marker at the close ──────────────────────────────────────────────
        ax.scatter(xi, y_event, color=color, marker=marker,
                   s=90, zorder=6, linewidths=0)

        # ── Horizontal dashed level line from break point rightward ──────────
        x_end = min(xi + 30, len(df) - 1)
        ax.hlines(e.level, xi, x_end,
                  colors=color, linewidth=0.9,
                  linestyles="--", alpha=LEVEL_ALPHA, zorder=4)

        # ── Price label (deduplicate crowded labels) ──────────────────────────
        price_key = f"{e.level:.0f}"
        too_close = (price_key in label_budget
                     and abs(xi - label_budget[price_key]) < 10)
        if not too_close:
            label_budget[price_key] = xi
            va = "bottom" if e.direction == "bullish" else "top"
            y_off = y_event * 0.003 * (1 if e.direction == "bullish" else -1)
            ax.annotate(
                f" {prefix}\n ${e.level:,.0f}",
                xy=(xi, y_event),
                xytext=(xi + 0.5, y_event + y_off),
                fontsize=7.5,
                color=color,
                va=va,
                ha="left",
                zorder=7,
                annotation_clip=True,
            )

    # ── X-axis: map integer positions back to dates ───────────────────────────
    # Tick every ~30 bars (roughly monthly on daily)
    step = max(1, len(df) // 20)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [str(df.index[i])[:10] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right",
                       color="#8b949e", fontsize=8)

    # ── Y-axis formatting ─────────────────────────────────────────────────────
    ax.yaxis.set_tick_params(colors="#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.set_ylabel("Price (USD)", color="#8b949e")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_xlim(-1, len(df))

    # ── Title ─────────────────────────────────────────────────────────────────
    n_bos   = sum(1 for e in visible_events if e.kind == "bos")
    n_choch = sum(1 for e in visible_events if e.kind == "choch")
    bias    = struct_res.bias()
    bias_color = BOS_BULL if bias == "bullish" else BOS_BEAR

    ax.set_title(
        f"BTC-USD  ({tf})  —  BOS / CHoCH  (N={n}, close-based)  "
        f"[{df.index[0].date()} - {df.index[-1].date()}]  |  "
        f"BOS: {n_bos}   CHoCH: {n_choch}   "
        f"Bias: {bias.upper()}",
        color="#e6edf3", fontsize=11, pad=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=BOS_BULL,
               markersize=9, label="BOS Bullish", linewidth=0),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=BOS_BEAR,
               markersize=9, label="BOS Bearish", linewidth=0),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=CHOCH_BULL,
               markersize=8, label="CHoCH Bullish", linewidth=0),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=CHOCH_BEAR,
               markersize=8, label="CHoCH Bearish", linewidth=0),
        mpatches.Patch(facecolor=BULL_CANDLE, label="Bull candle"),
        mpatches.Patch(facecolor=BEAR_CANDLE, label="Bear candle"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", ncol=3)

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    plt.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor=BG)
    plt.close()
    print(f"Chart saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="BOS/CHoCH price chart for BTC-USD")
    parser.add_argument("--n",    type=int, default=7, choices=valid_n_values())
    parser.add_argument("--tf",   type=str, default="1d", choices=list(TIMEFRAME_MAP))
    parser.add_argument("--bars", type=int, default=365,
                        help="Number of most-recent bars to display (default 365)")
    args = parser.parse_args()

    df_full = _fetch("BTC-USD", tf=args.tf)

    n_bars = min(args.bars, len(df_full))
    out = (Path(__file__).parent.parent / "charts"
           / f"bos_choch_{args.tf}_n{args.n}_{n_bars}bars.png")

    build_chart(df_full, n=args.n, n_bars=n_bars, tf=args.tf, out_path=out)


if __name__ == "__main__":
    main()
