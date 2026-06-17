"""
strategy_breakout/signal.py
===========================
Session-open breakout + retest entry detection on H1 bars.

Sessions (UTC):
  Asia   00-08  entry window: 00-03
  London 08-16  entry window: 08-11
  NY     13-21  entry window: 13-16

S/R levels:
  London  <- prior Asia H/L  (same calendar day)
  NY      <- London 08-12 H/L only  (pre-NY, no look-ahead)
  Asia    <- prior day combined H/L

Entry conditions:
  1. Bar is inside the session entry window
  2. Volume > 1.5x 20-bar rolling average
  3. Prior close was on one side of the S/R level
  4. Current close breaks through the level (breakout bar)
  5. Next 1-8 bars retest the broken level within 0.35 ATR

All bugs from exit_optimizer audit are implemented correctly here.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

SESSIONS = {
    "Asia":   (0,  8),
    "London": (8,  16),
    "NY":     (13, 21),
}
ENTRY_WINDOW         = 4    # hours after session open accepted for breakout
LONDON_FOR_NY_END    = 13   # London bars used for NY S/R end at 13:00 (exclusive)
ATR_PERIOD           = 14
VOLUME_MULT          = 1.5
STOP_ATR_MULT        = 0.4
MAX_RETEST_BARS      = 8
MAX_TRADES_PER_SESS  = 1


@dataclass
class BreakoutSignal:
    symbol:    str
    direction: str        # "long" | "short"
    entry_bar: int        # index in df where entry (retest) bar is
    bo_bar:    int        # index of the breakout bar
    level:     float      # broken S/R level
    entry_px:  float      # level price (spread added in loop)
    stop:      float      # initial stop loss
    risk:      float      # abs(entry_px - stop)
    atr:       float
    session:   str
    bar_time:  pd.Timestamp


# ── Indicators ────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, p: int = ATR_PERIOD) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def _avg_vol(df: pd.DataFrame, p: int = 20) -> pd.Series:
    return df["volume"].rolling(p, min_periods=5).mean()


# ── Session H/L table ─────────────────────────────────────────────────────────

def build_session_hl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by (date, session) with columns H and L.
    Includes 'London_preNY' (08-12 bars only) for look-ahead-free NY S/R.
    """
    tmp = df[["high", "low"]].copy()
    tmp["date"] = df.index.date
    tmp["hour"] = df.index.hour

    records = []

    for sess, (o, c) in SESSIONS.items():
        mask = (tmp["hour"] >= o) & (tmp["hour"] < c)
        g = tmp[mask].groupby("date").agg(H=("high", "max"), L=("low", "min"))
        g["session"] = sess
        records.append(g.reset_index())

    # London pre-NY: only bars 08:00-12:59
    mask = (tmp["hour"] >= 8) & (tmp["hour"] < LONDON_FOR_NY_END)
    g = tmp[mask].groupby("date").agg(H=("high", "max"), L=("low", "min"))
    g["session"] = "London_preNY"
    records.append(g.reset_index())

    hl = pd.concat(records, ignore_index=True)
    hl["date"] = pd.to_datetime(hl["date"])
    return hl.set_index(["date", "session"])


def attach_sr(df: pd.DataFrame, hl: pd.DataFrame) -> pd.DataFrame:
    """
    Attach resistance and support to bars inside entry windows.
    NY session uses London_preNY S/R (look-ahead free).
    """
    out   = df.copy()
    hours = df.index.hour
    dates = pd.to_datetime(df.index.date)
    flat  = hl.reset_index()

    out["resistance"] = np.nan
    out["support"]    = np.nan
    out["session"]    = ""

    def _join(sess_name: str, src_sess: str, day_shift: int = 0):
        o = SESSIONS[sess_name][0]
        mask = (hours >= o) & (hours < o + ENTRY_WINDOW)
        idx  = df.index[mask]
        if idx.empty:
            return

        bars = pd.DataFrame(
            {"date": pd.to_datetime(df.index[mask].date)},
            index=idx,
        )
        if day_shift:
            bars["date"] = bars["date"] + pd.Timedelta(days=day_shift)

        src = flat[flat["session"] == src_sess][["date", "H", "L"]].copy()
        src["date"] = pd.to_datetime(src["date"])

        merged = bars.merge(src, on="date", how="left")
        merged.index = idx

        out.loc[idx, "resistance"] = merged["H"].values
        out.loc[idx, "support"]    = merged["L"].values
        out.loc[idx, "session"]    = sess_name

    _join("London", "Asia")
    _join("NY",     "London_preNY")      # bug-1 fix: pre-NY London only

    # Asia: use combined prior-day H/L (all sessions from prior calendar day)
    ao   = SESSIONS["Asia"][0]
    mask = (hours >= ao) & (hours < ao + ENTRY_WINDOW)
    idx  = df.index[mask]
    if not idx.empty:
        prev_hl = flat.groupby("date").agg(H=("H", "max"), L=("L", "min")).reset_index()
        prev_hl["date"] = pd.to_datetime(prev_hl["date"])
        bars = pd.DataFrame({"date": pd.to_datetime(idx.date)}, index=idx)
        bars["date"] = bars["date"] - pd.Timedelta(days=1)
        merged = bars.merge(prev_hl, on="date", how="left")
        merged.index = idx
        out.loc[idx, "resistance"] = merged["H"].values
        out.loc[idx, "support"]    = merged["L"].values
        out.loc[idx, "session"]    = "Asia"

    return out


# ── Entry scanner ─────────────────────────────────────────────────────────────

def find_entries(df: pd.DataFrame, symbol: str) -> list[BreakoutSignal]:
    """
    Scan a completed H1 bar history for breakout + retest entry signals.
    Returns list of BreakoutSignal — each is one potential trade entry.

    Designed for backtest use.  Live use calls find_latest_entry() instead.
    """
    df = df.copy()
    df["atr"]     = _atr(df)
    df["avg_vol"] = _avg_vol(df)
    hl  = build_session_hl(df)
    df  = attach_sr(df, hl)

    close  = df["close"]
    prev_c = close.shift(1)
    vol    = df["volume"]
    avg_v  = df["avg_vol"]
    res    = df["resistance"]
    sup    = df["support"]
    in_s   = df["session"] != ""
    vol_ok = vol > VOLUME_MULT * avg_v

    bo_up = in_s & vol_ok & prev_c.notna() & res.notna() & (prev_c <= res) & (close > res)
    bo_dn = in_s & vol_ok & prev_c.notna() & sup.notna()  & (prev_c >= sup) & (close < sup)

    events = (
        [(i, "long",  float(res.iloc[i])) for i in np.where(bo_up.values)[0]] +
        [(i, "short", float(sup.iloc[i])) for i in np.where(bo_dn.values)[0]]
    )
    events.sort(key=lambda x: x[0])

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values
    n      = len(df)

    signals    = []
    skip_until = 0
    sess_count: dict = {}

    for bo_i, direction, level in events:
        if bo_i < skip_until or bo_i + 1 >= n or np.isnan(level):
            continue

        atr_v = atrs[bo_i]
        if np.isnan(atr_v) or atr_v <= 0:
            continue

        sess = df["session"].iloc[bo_i]
        date = df.index[bo_i].date()
        key  = (date, sess)
        if sess_count.get(key, 0) >= MAX_TRADES_PER_SESS:
            continue

        tol      = atr_v * 0.35
        entry_i  = None
        for j in range(bo_i + 1, min(bo_i + MAX_RETEST_BARS + 1, n)):
            if direction == "long"  and lows[j]  <= level + tol:
                entry_i = j; break
            if direction == "short" and highs[j] >= level - tol:
                entry_i = j; break

        if entry_i is None:
            continue

        stop = (level - STOP_ATR_MULT * atr_v if direction == "long"
                else level + STOP_ATR_MULT * atr_v)
        risk = abs(level - stop)
        if risk <= 0:
            continue

        signals.append(BreakoutSignal(
            symbol    = symbol,
            direction = direction,
            entry_bar = entry_i,
            bo_bar    = bo_i,
            level     = level,
            entry_px  = level,
            stop      = stop,
            risk      = risk,
            atr       = atr_v,
            session   = sess,
            bar_time  = df.index[entry_i],
        ))

        sess_count[key] = sess_count.get(key, 0) + 1
        skip_until = bo_i + MAX_RETEST_BARS + 1

    return signals


def find_latest_entry(
    df: pd.DataFrame,
    symbol: str,
    last_acted_ts: Optional[pd.Timestamp],
) -> Optional[BreakoutSignal]:
    """
    Live-mode entry check: scan recent H1 bars and return the freshest
    signal not yet acted upon.  Returns None if no new signal.
    """
    signals = find_entries(df, symbol)
    if not signals:
        return None

    latest = signals[-1]

    if last_acted_ts is not None and latest.bar_time <= last_acted_ts:
        return None

    return latest


def session_end_bar(df: pd.DataFrame, session: str, entry_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Return the timestamp of the last H1 bar in the given session on the
    same calendar day as entry_ts.
    """
    o, c   = SESSIONS[session]
    date   = entry_ts.date()
    mask   = (
        (df.index.date == date) &
        (df.index.hour >= o)    &
        (df.index.hour <  c)
    )
    bars = df.index[mask]
    return bars[-1] if len(bars) > 0 else None
