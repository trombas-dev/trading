"""
mt5_price.py — Live multi-timeframe bar fetcher for the Hermes MTF loop.

Sources (tried in order)
------------------------
  Primary : MetaTrader 5 Python API  (requires MT5 terminal + ICMarkets account)
  Fallback : yfinance                 (development / CI; limited history)

MT5 is synchronous; all heavy work is dispatched to a thread pool via
asyncio.to_thread() so the event loop is never blocked.

All three scenarios are supported:
  Scenario 1 (default) : 1H bias → 15M signal → 5M entry
  Scenario 2           : 4H bias → 1H signal  → 15M entry
  Scenario 3           : 1D bias → 4H signal  → 1H entry

MT5 has native 4H and 1D timeframes.
yfinance 4H is synthesised by downloading 1H bars and resampling.
yfinance 1D is downloaded directly with interval="1d".

Public API
----------
  async fetch_mtf_bars(symbol, source, scenario, n_bias, n_signal, n_entry)
      -> dict[str, pd.DataFrame]   keyed by TF string, UTC-indexed OHLCV

  SYMBOL_MAP
      MT5 symbol name -> yfinance ticker  (for fallback)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# MT5 is single-threaded internally; serialise all initialize/shutdown calls.
_MT5_LOCK: asyncio.Lock | None = None


def _get_mt5_lock() -> asyncio.Lock:
    """Return the per-event-loop MT5 lock, creating it lazily."""
    global _MT5_LOCK
    if _MT5_LOCK is None:
        _MT5_LOCK = asyncio.Lock()
    return _MT5_LOCK


def _load_env() -> None:
    """Load .env from the project root (best-effort; no hard dependency on python-dotenv)."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if key and key not in os.environ:   # don't override already-set vars
            os.environ[key] = val


_load_env()   # run once at import time


def _mt5_credentials() -> tuple[str, int, str]:
    """Read MT5 credentials from environment. Raises if any are missing."""
    server   = os.environ.get("MT5_SERVER", "")
    login    = os.environ.get("MT5_LOGIN",  "")
    password = os.environ.get("MT5_PASSWORD", "")
    if not server or not login or not password:
        raise EnvironmentError(
            "MT5 credentials not set. Add MT5_SERVER, MT5_LOGIN, MT5_PASSWORD to .env"
        )
    return server, int(login), password


# ── Scenario TF triplets (mirrors backtest.py to avoid circular import) ───────

_SCENARIO_TFS: dict[int, tuple[str, str, str]] = {
    1: ("1H",  "15M", "5M"),
    2: ("4H",  "1H",  "15M"),
    3: ("1D",  "4H",  "1H"),
}

# ── Symbol map ────────────────────────────────────────────────────────────────

SYMBOL_MAP: dict[str, str] = {
    # Forex
    "BTCUSD":  "BTC-USD",
    "EURUSD":  "EURUSD=X",
    "GBPUSD":  "GBPUSD=X",
    "USDJPY":  "USDJPY=X",
    "NZDUSD":  "NZDUSD=X",
    "EURGBP":  "EURGBP=X",
    "USDCHF":  "USDCHF=X",
    "GBPCHF":  "GBPCHF=X",
    "CHFJPY":  "CHFJPY=X",
    "GBPAUD":  "GBPAUD=X",
    "GBPJPY":  "GBPJPY=X",
    "EURJPY":  "EURJPY=X",
    # Commodities
    "XAUUSD":  "GC=F",
    "XAGUSD":  "SI=F",
    "XTIUSD":  "CL=F",
    "XBRUSD":  "BZ=F",
    "XNGUSD":  "NG=F",
    # Indices
    "US30":    "^DJI",
    "US500":   "^GSPC",
    "JP225":   "^N225",
    "AUS200":  "^AXJO",
    "UK100":   "^FTSE",
    "GER40":   "^GDAXI",
}

_MT5_TF: dict[str, str] = {
    "1D":  "TIMEFRAME_D1",
    "4H":  "TIMEFRAME_H4",
    "1H":  "TIMEFRAME_H1",
    "15M": "TIMEFRAME_M15",
    "5M":  "TIMEFRAME_M5",
}

# yfinance: 4H has no native support — fetch 1H and resample.
# 1D:       fetch directly.
_YF_INTERVAL: dict[str, str] = {
    "1D":  "1d",
    "4H":  "1h",    # fetch 1H, resample -> 4H
    "1H":  "1h",
    "15M": "15m",
    "5M":  "5m",
}

_YF_PERIOD: dict[str, str] = {
    "1D":  "5y",    # yfinance multi-year daily
    "4H":  "730d",  # fetch 1H then resample
    "1H":  "730d",
    "15M": "60d",
    "5M":  "60d",
}


# ── Spread fetcher ────────────────────────────────────────────────────────────

def _fetch_half_spread_sync(symbol: str, extra_pts: int = 3) -> float:
    """
    Return half the round-trip transaction cost in price units.

    Cost = (ask - bid) + extra_pts * point_size
    Half-spread = cost / 2  (applied once on entry, once on exit)

    Parameters
    ----------
    symbol     : MT5 symbol name.
    extra_pts  : Slippage buffer in points added on top of the raw spread.
                 Default 3 points to cover typical execution slippage.

    Returns 0.0 on any MT5 failure so the caller can treat it as zero cost.
    """
    import MetaTrader5 as mt5

    server, login, password = _mt5_credentials()
    if not mt5.initialize(server=server, login=login, password=password):
        logger.warning(f"MT5 initialize() failed for spread fetch: {mt5.last_error()}")
        return 0.0

    try:
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None or info.point == 0:
            return 0.0
        raw_spread = tick.ask - tick.bid          # live bid-ask spread in price
        buffer     = extra_pts * info.point       # slippage buffer
        return (raw_spread + buffer) / 2.0        # half the round-trip cost
    finally:
        mt5.shutdown()


async def fetch_half_spread(
    symbol: str,
    source: str = "mt5",
    extra_pts: int = 3,
) -> float:
    """
    Async wrapper: fetch current half-spread + slippage buffer.

    source="postgres" : reads spread pushed by the Windows MT5 bridge
    source="mt5"      : reads directly from MT5 terminal (Windows only)
    source="yfinance" : returns 0.0 (no live tick data available)
    """
    if source == "postgres":
        spread = await asyncio.to_thread(_pg_spread_sync, symbol)
        if spread is not None:
            return spread / 2.0
        # Fall through to 0.0 — bridge may not have run yet
        return 0.0

    if source != "mt5":
        return 0.0
    try:
        lock = _get_mt5_lock()
        async with lock:
            return await asyncio.to_thread(
                _fetch_half_spread_sync, symbol, extra_pts
            )
    except Exception as exc:
        logger.warning(f"fetch_half_spread({symbol}) failed: {exc}")
        return 0.0


# ── Column normalisation ──────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, UTC DatetimeIndex, keep OHLCV only."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0].lower() if isinstance(c, tuple) else c.lower()
            for c in df.columns
        ]
    else:
        df.columns = [c.lower() for c in df.columns]

    rename = {
        "vol": "volume", "tick_volume": "volume", "real_volume": "volume",
    }
    df.rename(columns=rename, inplace=True)

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing OHLC column '{col}'. Got: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a lower-TF DataFrame to a higher TF (e.g. 1H -> 4H)."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


# ── MT5 (synchronous) ─────────────────────────────────────────────────────────

def _mt5_bars_sync(symbol: str, tf: str, n_bars: int) -> pd.DataFrame:
    """Fetch the last `n_bars` from MT5 terminal (sync, called in thread)."""
    import MetaTrader5 as mt5

    server, login, password = _mt5_credentials()
    if not mt5.initialize(server=server, login=login, password=password):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    try:
        # Ensure the symbol is visible in Market Watch so intraday data is
        # available.  This is a no-op if the symbol is already subscribed.
        mt5.symbol_select(symbol, True)

        tf_const = getattr(mt5, _MT5_TF[tf])
        rates    = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
    finally:
        mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"MT5 returned no data for {symbol} {tf}")

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("timestamp", inplace=True)
    df.drop(columns=["time", "spread"], errors="ignore", inplace=True)
    return _normalise(df)


# ── yfinance (synchronous) ────────────────────────────────────────────────────

def _yf_bars_sync(symbol: str, tf: str, n_bars: int) -> pd.DataFrame:
    """Fetch bars via yfinance (sync, called in thread).

    4H bars: yfinance has no native 4H interval, so we download 1H and
    resample.  All other timeframes are fetched directly.
    """
    import yfinance as yf

    ticker = SYMBOL_MAP.get(symbol)
    if not ticker:
        raise ValueError(f"No yfinance ticker mapping for '{symbol}'")

    # For 4H we fetch 1H and resample
    fetch_tf = "1H" if tf == "4H" else tf
    interval = _YF_INTERVAL[fetch_tf]
    period   = _YF_PERIOD[tf]

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned empty data for {ticker} / {interval}")

    df = _normalise(raw)

    if tf == "4H":
        df = _resample(df, "4h")

    # Trim to requested bar count (take the last n_bars)
    if len(df) > n_bars:
        df = df.iloc[-n_bars:]

    return df


# ── PostgreSQL bridge (synchronous) ───────────────────────────────────────────

def _pg_bars_sync(symbol: str, tf: str, n_bars: int) -> pd.DataFrame:
    """Read bars from the Railway PostgreSQL bridge table (sync, called in thread)."""
    import psycopg2
    import psycopg2.extras

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set — cannot use postgres source")

    conn = psycopg2.connect(db_url, sslmode="require")
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM bars
                WHERE symbol = %s AND timeframe = %s
                ORDER BY ts DESC
                LIMIT %s
                """,
                (symbol, tf, n_bars),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"No Postgres bars for {symbol} {tf}")

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.set_index("ts", inplace=True)
    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    return _normalise(df)


def _pg_spread_sync(symbol: str) -> float | None:
    """Read the latest spread from the spreads table. Returns None if unavailable."""
    try:
        import psycopg2

        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            return None
        conn = psycopg2.connect(db_url, sslmode="require")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT spread, updated_at FROM spreads WHERE symbol = %s",
                    (symbol,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        spread, updated_at = row
        # Ignore stale spreads (older than 30 minutes)
        from datetime import datetime, timezone, timedelta
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - updated_at
        if age > timedelta(minutes=30):
            return None
        return float(spread)
    except Exception as exc:
        logger.warning(f"pg_spread({symbol}) failed: {exc}")
        return None


# ── Async wrappers ────────────────────────────────────────────────────────────

async def _fetch_bars(symbol: str, tf: str, n_bars: int, source: str) -> pd.DataFrame:
    """Fetch bars for one timeframe; falls back down the source chain.

    Source priority:
      "mt5"      -> MT5 terminal (Windows only, serialised via lock)
      "postgres" -> Railway Postgres bridge table
      "yfinance" -> yfinance (public data, always available)

    When source="mt5" or "postgres", automatic fallback to yfinance on failure.
    """
    if source == "postgres":
        try:
            return await asyncio.to_thread(_pg_bars_sync, symbol, tf, n_bars)
        except Exception as exc:
            logger.warning(
                f"Postgres fetch failed for {symbol} {tf}: {exc}. "
                f"Falling back to yfinance."
            )
        return await asyncio.to_thread(_yf_bars_sync, symbol, tf, n_bars)

    if source == "mt5":
        try:
            lock = _get_mt5_lock()
            async with lock:
                return await asyncio.to_thread(_mt5_bars_sync, symbol, tf, n_bars)
        except Exception as exc:
            logger.warning(
                f"MT5 fetch failed for {symbol} {tf}: {exc}. "
                f"Falling back to yfinance."
            )

    # yfinance path (explicit or fallback)
    return await asyncio.to_thread(_yf_bars_sync, symbol, tf, n_bars)


async def fetch_mtf_bars(
    symbol:    str,
    source:    str = "yfinance",
    scenario:  int = 1,
    n_bias:    int = 1000,
    n_signal:  int = 500,
    n_entry:   int = 500,
    extra_tfs: "dict[str, int] | None" = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch the three timeframes required for the active scenario, plus any extras.

    Parameters
    ----------
    symbol    : MT5 symbol name (e.g. "BTCUSD", "EURUSD").
    source    : "mt5" (requires terminal) | "yfinance" (default fallback).
    scenario  : 1=1H/15M/5M  2=4H/1H/15M  3=1D/4H/1H
    n_bias    : Number of bias-TF bars to fetch.
    n_signal  : Number of signal-TF bars to fetch.
    n_entry   : Number of entry-TF bars to fetch.
    extra_tfs : Optional {tf_string: n_bars} for additional timeframes needed
                by the regime filter (e.g. {"1D": 250} for ADX on daily).
                TFs already covered by the scenario are deduplicated.

    Returns
    -------
    dict[str, pd.DataFrame]  — keys are TF strings, values are UTC-indexed
    OHLCV DataFrames.  E.g. {"1H": df_1h, "15M": df_15m, "5M": df_5m}.

    Raises
    ------
    RuntimeError if all sources fail for any required timeframe.
    """
    tf_bias, tf_sig, tf_ent = _SCENARIO_TFS.get(scenario, _SCENARIO_TFS[1])

    # Build the full list of (tf, n_bars) to fetch, deduplicating by TF
    fetch_plan: dict[str, int] = {
        tf_bias: n_bias,
        tf_sig:  n_signal,
        tf_ent:  n_entry,
    }
    if extra_tfs:
        for tf, n in extra_tfs.items():
            if tf not in fetch_plan:   # don't override scenario TFs
                fetch_plan[tf] = n

    tfs_ordered = list(fetch_plan.keys())
    results = await asyncio.gather(
        *[_fetch_bars(symbol, tf, fetch_plan[tf], source) for tf in tfs_ordered]
    )
    dfs = dict(zip(tfs_ordered, results))

    logger.debug(
        f"fetch_mtf_bars {symbol} s{scenario}: "
        + "  ".join(f"{tf}={len(df)}" for tf, df in dfs.items())
        + " bars"
    )
    return dfs
