"""
compare_default_vs_optimised.py
--------------------------------
Runs default vs optimised params for all 13 pairs (full history 2019-2026).
Profit calculated at 1% risk per trade on a $10,000 account (non-compounding).
"""
import dataclasses
import io
import sys

# Force UTF-8 on Windows consoles (cp1257 / cp1252 don't support box-draw chars)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from hermes_trading.strategy.backtest import BacktestParams, run_single_backtest, compute_metrics

ACCOUNT  = 10_000   # starting account $
RISK_PCT = 0.01     # 1% risk per trade

# ── Spread tables ─────────────────────────────────────────────────────────────
# half_spread = spread_pts * point_size / 2  (half of round-trip cost, in price units)
# Applied once on entry and once on exit inside the backtest engine.
# spread_pts: round-trip spread in instrument points (5-digit broker).

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
# Spread = (ask - bid) / point_size  — the raw market cost, no extra buffer.
_SPREAD_PTS: dict[str, int] = {
    "EURUSD":  9,  "GBPUSD": 12,  "NZDUSD": 11,
    "USDCHF":  9,  "EURGBP": 13,  "GBPAUD": 18,  "GBPCHF": 15,
    "AUDUSD": 10,  "USDCAD": 10,
    "USDJPY": 10,  "EURJPY": 21,  "GBPJPY": 22,  "CHFJPY": 16,
    "XAUUSD": 19,  "BTCUSD": 1200,
}


def _half_spread(sym: str) -> float:
    """Convert spread_pts -> half_spread in price units for a symbol."""
    pts = _SPREAD_PTS.get(sym, 20)
    ps  = _POINT_SIZE.get(sym, 0.00001)
    return pts * ps / 2

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "EURJPY", "NZDUSD", "XAUUSD", "BTCUSD",
    "EURGBP", "USDCHF", "GBPCHF", "CHFJPY", "GBPAUD",
]

# ── Default params ────────────────────────────────────────────────────────────
DEFAULT = BacktestParams(
    n_bias=9, n_signal=3, n_entry=3,
    zone_lo=0.236, zone_hi=0.618,
    timeout_bars=6, max_bars_in_trade=100,
)

# ── Optimised params per symbol (from grid search IS 2019-2022) ───────────────
# Rules applied:
#   GBPUSD nb5:         +32R -> +134R, Sharpe 0.42 -> 1.83 — clear improvement
#   GBPJPY nb5+zhi+t2:  +31R ->  +58R, MaxDD 41.6 -> 13.1R — clear improvement
#   USDJPY BOS mode:    +59R ->  +27R  (-31R)               — reverted to default
#   NZDUSD ns5+tighter: +113R -> +51R  (-62R)               — reverted to default
#   EURJPY nb5+t2:      -4.67R marginal                     — reverted to default
OPTIMISED = {
    "EURUSD": DEFAULT,  # grid winner = default params
    "GBPUSD": BacktestParams(          # nb5 wins for GBP erratic structure (+102R improvement)
        n_bias=5, n_signal=3, n_entry=3,
        zone_lo=0.236, zone_hi=0.618,
        timeout_bars=6, max_bars_in_trade=100,
    ),
    "USDJPY": DEFAULT,  # BOS mode net -31R on full history — default wins
    "GBPJPY": BacktestParams(          # nb5 + wider zone + fast timeout (+27R, MaxDD 41->13)
        n_bias=5, n_signal=3, n_entry=3,
        zone_lo=0.236, zone_hi=0.786,
        timeout_bars=2, max_bars_in_trade=100,
    ),
    "EURJPY": DEFAULT,  # nb5+t2 only -4.67R marginal — keep default
    "NZDUSD": DEFAULT,  # ns5+tighter only 45 IS trades, -62R on full history — default wins
    "XAUUSD": DEFAULT,  # grid winner = default params
    "BTCUSD": DEFAULT,  # grid winner = default params
    # New pairs — use default until grid optimizer run
    "EURGBP": DEFAULT,
    "USDCHF": DEFAULT,
    "GBPCHF": DEFAULT,
    "CHFJPY": DEFAULT,
    "GBPAUD": DEFAULT,
}

def load(sym):
    return {tf: pd.read_parquet(f"data/{sym}/{tf}.parquet", engine="pyarrow")
            for tf in ("1D", "4H", "1H", "15M", "5M")}

def profit_usd(total_r):
    """Non-compounding profit at 1% risk on $10k account."""
    return total_r * RISK_PCT * ACCOUNT

# ── Run all backtests ─────────────────────────────────────────────────────────
print("Running backtests… ", end="", flush=True)
results = {}
for sym in PAIRS:
    try:
        dfs = load(sym)
    except FileNotFoundError:
        print(f"\n  SKIP {sym} (no parquet)")
        continue

    hs = _half_spread(sym)
    p_def_sym = dataclasses.replace(DEFAULT,        half_spread=hs)
    p_opt     = OPTIMISED[sym]
    is_same   = (p_opt is DEFAULT)
    p_opt_sym = dataclasses.replace(p_opt,          half_spread=hs)

    m_def = compute_metrics(run_single_backtest(dfs, params=p_def_sym))
    m_opt = m_def if is_same else compute_metrics(run_single_backtest(dfs, params=p_opt_sym))

    results[sym] = (m_def, m_opt, is_same)
    print(sym[0], end="", flush=True)

print(" done\n")

# ── Print comparison table ────────────────────────────────────────────────────
SEP  = "-" * 115
HDR1 = f"{'':8s}  {'-- DEFAULT (nb9 ns3 z0.24-0.62 t6) --------------------------------------------------':54s}  {'-- OPTIMISED ---------------------------------------------------':50s}"
HDR2 = (f"{'Symbol':8s}  {'Trades':>6}  {'WR%':>5}  {'Total R':>8}  {'Profit $':>9}  {'Sharpe':>6}  {'MaxDD':>6}  "
        f"{'Trades':>6}  {'WR%':>5}  {'Total R':>8}  {'Profit $':>9}  {'Sharpe':>6}  {'MaxDD':>6}  {'dR':>7}  {'Opt params'}")

print(SEP)
print(HDR1)
print(HDR2)
print(SEP)

for sym in PAIRS:
    if sym not in results:
        continue
    m_d, m_o, same = results[sym]

    def fmt(m):
        n  = m.get("n_trades", 0)
        wr = m.get("win_rate", 0)
        tr = m.get("total_r",  0)
        sh = m.get("sharpe",   0)
        dd = m.get("max_dd_r", 0)
        pf = profit_usd(tr)
        return n, wr, tr, pf, sh, dd

    dn, dwr, dtr, dpf, dsh, ddd = fmt(m_d)
    on, owr, otr, opf, osh, odd = fmt(m_o)
    delta = otr - dtr
    d_sign = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
    marker = "  " if same else "* "

    p_opt = OPTIMISED[sym]
    if same:
        opt_label = "(same as default)"
    else:
        parts = []
        if p_opt.n_bias    != DEFAULT.n_bias:    parts.append(f"nb{p_opt.n_bias}")
        if p_opt.n_signal  != DEFAULT.n_signal:  parts.append(f"ns{p_opt.n_signal}")
        if p_opt.zone_lo   != DEFAULT.zone_lo:   parts.append(f"zlo{p_opt.zone_lo:.3f}")
        if p_opt.zone_hi   != DEFAULT.zone_hi:   parts.append(f"zhi{p_opt.zone_hi:.3f}")
        if p_opt.timeout_bars != DEFAULT.timeout_bars: parts.append(f"t{p_opt.timeout_bars}")
        if p_opt.use_bos_windows: parts.append("BOS")
        opt_label = marker + " ".join(parts)

    print(
        f"{sym:8s}  "
        f"{dn:6d}  {dwr:5.1f}  {dtr:+8.2f}  ${dpf:8.0f}  {dsh:6.2f}  {ddd:6.2f}  "
        f"{on:6d}  {owr:5.1f}  {otr:+8.2f}  ${opf:8.0f}  {osh:6.2f}  {odd:6.2f}  "
        f"{d_sign:>7}  {opt_label}"
    )

print(SEP)

# ── Totals ────────────────────────────────────────────────────────────────────
tot_d_r  = sum(results[s][0].get("total_r", 0) for s in results)
tot_o_r  = sum(results[s][1].get("total_r", 0) for s in results)
tot_d_tr = sum(results[s][0].get("n_trades", 0) for s in results)
tot_o_tr = sum(results[s][1].get("n_trades", 0) for s in results)
tot_delta = tot_o_r - tot_d_r

tot_d_sign = f"+{tot_delta:.2f}" if tot_delta >= 0 else f"{tot_delta:.2f}"
print(
    f"{'TOTAL':8s}  "
    f"{tot_d_tr:6d}  {'':5s}  {tot_d_r:+8.2f}  ${profit_usd(tot_d_r):8.0f}  {'':6s}  {'':6s}  "
    f"{tot_o_tr:6d}  {'':5s}  {tot_o_r:+8.2f}  ${profit_usd(tot_o_r):8.0f}  {'':6s}  {'':6s}  "
    f"{tot_d_sign:>7}"
)
print(SEP)
print(f"\n  Account: ${ACCOUNT:,.0f}  |  Risk per trade: {RISK_PCT*100:.0f}%  |  Profit = Total R × ${RISK_PCT*ACCOUNT:.0f}")
print(f"  [!] NZDUSD optimised: only 45 IS / 16 OOS trades -- treat with caution")
