"""
regime.py — Higher-timeframe regime detection for the Hermes MTF strategy.

A regime filter evaluates whether the market is in a TRENDING or RANGING/CHOPPY
state on a higher timeframe.  It is applied at the WINDOW level in backtest.py
and live.py: windows that open during a choppy regime are skipped entirely.

Why window-level, not bar-level?
---------------------------------
Applying the filter when a CHoCH opens a window (rather than on every 5M bar)
keeps the logic clean and avoids interfering with entries/exits inside an
already-open window.  If the market was trending when the window opened, we run
the full signal → entry → exit pipeline as usual.

Available filters
-----------------
  "none"       No filter — all windows allowed (default, matches prior results).
  "adx"        ADX(period) on the regime TF.  Skip when ADX < adx_min (20).
               ADX measures trend STRENGTH (not direction) — a pure chop filter.
               Wilder's canonical thresholds: <20 = no trend, 20-25 = weak trend,
               >25 = trending.  Default 20.0 is the conservative minimum.
  "atr_ratio"  ATR(short) / ATR(long) on the regime TF.
               Low ratio → volatility compression → likely range-bound.
               Skip when ratio < atr_ratio_min (default 0.80).

Both indicators use Wilder's smoothing (alpha = 1/period), matching standard
MT5 / TradingView ADX behaviour.

Key functions
-------------
  compute_regime_ok(df, filter_type, **kwargs) -> pd.Series[bool]
      Returns True where trading is allowed (trending regime), False for choppy.
      NaN positions (warm-up period) default to True (fail open — no data → allow).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Wilder's EMA ──────────────────────────────────────────────────────────────

def _wilder_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's EMA: alpha = 1/period.

    Initialises with the simple mean of the first `period` non-NaN values,
    then applies:  out[i] = out[i-1] + (1/period) * (arr[i] - out[i-1])

    NaN inputs hold the previous value (no gap pollution).
    Returns NaN for all indices before initialisation.
    """
    n   = len(arr)
    out = np.full(n, np.nan)

    count    = 0
    total    = 0.0
    init_idx = -1

    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            continue
        total += v
        count += 1
        if count == period:
            out[i] = total / period
            init_idx = i
            break

    if init_idx < 0:
        return out  # insufficient data

    alpha = 1.0 / period
    prev  = out[init_idx]
    for i in range(init_idx + 1, n):
        v = arr[i]
        if not np.isnan(v):
            prev = prev + alpha * (v - prev)
        out[i] = prev

    return out


# ── ATR ───────────────────────────────────────────────────────────────────────

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder's EMA of True Range).

    True Range = max(H-L, |H-prev_C|, |L-prev_C|).
    TR[0] is undefined (no prior close); initialisation uses bars 1..period.

    Parameters
    ----------
    df     : OHLC DataFrame (columns: open, high, low, close — lower-case).
    period : Smoothing period (default 14).

    Returns
    -------
    pd.Series aligned to df.index, name = f"ATR_{period}".
    """
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    n      = len(highs)

    tr = np.full(n, np.nan)
    for i in range(1, n):
        hl   = highs[i] - lows[i]
        hpc  = abs(highs[i] - closes[i - 1])
        lpc  = abs(lows[i]  - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

    atr = _wilder_ema(tr, period)
    return pd.Series(atr, index=df.index, name=f"ATR_{period}")


# ── ADX ───────────────────────────────────────────────────────────────────────

def adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder's ADX).  Range: 0 – 100.

    Threshold guidance (Wilder's original):
      ADX < 20  → no trend (choppy / avoid trading)
      ADX 20-25 → weak trend forming
      ADX > 25  → trending / follow trend trades viable

    Computation (matches standard MT5 / TradingView ADX):
      1. True Range (TR), Directional Movement + (DM+), DM-
      2. Wilder EMA of each → ema_tr, ema_dmp, ema_dmm
      3. DI+ = 100 × ema_dmp / ema_tr
         DI- = 100 × ema_dmm / ema_tr
      4. DX  = 100 × |DI+ - DI-| / (DI+ + DI-)
      5. ADX = Wilder EMA of DX   (first valid ≈ 2 × period bars in)

    Parameters
    ----------
    df     : OHLC DataFrame.
    period : ADX period (default 14).

    Returns
    -------
    pd.Series aligned to df.index, name = f"ADX_{period}".
    """
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    n      = len(highs)

    tr       = np.full(n, np.nan)
    dm_plus  = np.full(n, np.nan)
    dm_minus = np.full(n, np.nan)

    for i in range(1, n):
        hl   = highs[i] - lows[i]
        hpc  = abs(highs[i] - closes[i - 1])
        lpc  = abs(lows[i]  - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up   = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        dm_plus[i]  = max(up,   0.0) if (up   > down and up   > 0) else 0.0
        dm_minus[i] = max(down, 0.0) if (down > up   and down > 0) else 0.0

    # Wilder EMA of each raw component
    ema_tr  = _wilder_ema(tr,       period)
    ema_dmp = _wilder_ema(dm_plus,  period)
    ema_dmm = _wilder_ema(dm_minus, period)

    # DI+ and DI- (suppress divide-by-zero warnings)
    with np.errstate(invalid="ignore", divide="ignore"):
        di_plus  = np.where(ema_tr > 0, 100.0 * ema_dmp / ema_tr, 0.0)
        di_minus = np.where(ema_tr > 0, 100.0 * ema_dmm / ema_tr, 0.0)

    # Propagate NaN from ema_tr (warm-up region)
    nan_mask = np.isnan(ema_tr)
    di_plus  = np.where(nan_mask, np.nan, di_plus)
    di_minus = np.where(nan_mask, np.nan, di_minus)

    # DX
    di_sum = di_plus + di_minus
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.where(di_sum > 0, 100.0 * np.abs(di_plus - di_minus) / di_sum, 0.0)
    dx = np.where(nan_mask, np.nan, dx)

    # ADX = Wilder EMA of DX
    adx = _wilder_ema(dx, period)
    return pd.Series(adx, index=df.index, name=f"ADX_{period}")


# ── Main public API ───────────────────────────────────────────────────────────

def compute_regime_ok(
    df: pd.DataFrame,
    filter_type: str,
    *,
    adx_period:    int   = 14,
    adx_min:       float = 20.0,
    atr_short:     int   = 10,
    atr_long:      int   = 50,
    atr_ratio_min: float = 0.80,
) -> pd.Series:
    """
    Compute a boolean Series: True = regime is trending (trade allowed).

    Parameters
    ----------
    df            : OHLC DataFrame on the chosen regime timeframe.
    filter_type   : "none" | "adx" | "atr_ratio"
    adx_period    : Period for ADX computation (default 14).
    adx_min       : Minimum ADX to allow trading (default 20.0).
    atr_short     : Short ATR period for atr_ratio filter (default 10).
    atr_long      : Long  ATR period for atr_ratio filter (default 50).
    atr_ratio_min : Minimum ATR_short/ATR_long ratio (default 0.80).

    Returns
    -------
    pd.Series[bool] aligned to df.index.
    NaN positions (warm-up period) are treated as True (fail open —
    insufficient data does not block trades).
    """
    if filter_type == "none":
        return pd.Series(True, index=df.index, dtype=bool)

    if filter_type == "adx":
        adx = adx_series(df, period=adx_period)
        ok  = adx >= adx_min
        # Warm-up NaN → True (fail open)
        ok  = ok.where(adx.notna(), other=True)
        return ok.astype(bool)

    if filter_type == "atr_ratio":
        atr_s = atr_series(df, period=atr_short)
        atr_l = atr_series(df, period=atr_long)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = atr_s / atr_l.replace(0.0, np.nan)
        ok = ratio >= atr_ratio_min
        # NaN (warm-up or zero denominator) → True (fail open)
        ok = ok.where(ratio.notna(), other=True)
        return ok.astype(bool)

    # Unknown filter type → fail open (don't silently block trades)
    return pd.Series(True, index=df.index, dtype=bool)
