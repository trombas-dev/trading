"""
download_mt5_data.py -- Historical OHLCV downloader for the MTF backtest.

Sources (tried in order)
------------------------
  1. MetaTrader 5 Python API  (primary -- needs MT5 terminal installed)
  2. yfinance                 (fallback -- limited history / some pairs unavailable)

Output
------
  data/
    <SYMBOL>/
      1H.parquet
      15M.parquet
      5M.parquet

Usage examples
--------------
  # All 10 pairs via MT5 (requires terminal open)
  uv run python scripts/download_mt5_data.py --source mt5

  # All pairs via yfinance fallback
  uv run python scripts/download_mt5_data.py --source yfinance

  # Single pair, MT5, custom date range
  uv run python scripts/download_mt5_data.py --symbol EURUSD --source mt5 --start 2019-01-01 --end 2025-12-31

  # List available symbols without downloading
  uv run python scripts/download_mt5_data.py --list
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
import io

import pandas as pd

# ── Load .env (no hard dependency on python-dotenv) ───────────────────────────
def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip(); val = val.strip()
        if key and key not in os.environ:
            os.environ[key] = val

_load_env()

# ── Symbol map ────────────────────────────────────────────────────────────────
# MT5 symbol name  ->  yfinance ticker (for fallback)

SYMBOL_MAP: dict[str, str] = {
    "BTCUSD":  "BTC-USD",
    "EURUSD":  "EURUSD=X",
    "GBPUSD":  "GBPUSD=X",
    "USDJPY":  "USDJPY=X",
    "XAUUSD":  "GC=F",        # Gold futures (best yf proxy)
    "GBPJPY":  "GBPJPY=X",
    "EURJPY":  "EURJPY=X",
    "NZDUSD":  "NZDUSD=X",
    "EURGBP":  "EURGBP=X",
    "USDCHF":  "USDCHF=X",
    "GBPCHF":  "GBPCHF=X",
    "CHFJPY":  "CHFJPY=X",
    "GBPAUD":  "GBPAUD=X",
}

# MT5 timeframe constants  (imported lazily to avoid hard dependency)
MT5_TF_MAP = {
    "1D":  "TIMEFRAME_D1",
    "4H":  "TIMEFRAME_H4",
    "1H":  "TIMEFRAME_H1",
    "15M": "TIMEFRAME_M15",
    "5M":  "TIMEFRAME_M5",
}

# yfinance: 4H and 1D are synthesised by resampling the 1H download.
# 1H yfinance is limited to ~730 days; 1D has full multi-year history.
YF_INTERVAL: dict[str, str] = {
    "1D":  "1d",
    "4H":  "1h",   # resample -> 4H inside _yf_download
    "1H":  "1h",
    "15M": "15m",
    "5M":  "5m",
}
YF_PERIOD: dict[str, str] = {
    "1D":  "10y",  # yfinance supports multi-year daily history
    "4H":  "2y",   # fetch 1H then resample
    "1H":  "2y",
    "15M": "60d",
    "5M":  "60d",
}

# Timeframes downloaded by default (all five)
TIMEFRAMES = ["1D", "4H", "1H", "15M", "5M"]
DATA_DIR   = Path(__file__).parent.parent / "data"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lower-case OHLCV columns regardless of source."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    rename = {"vol": "volume", "tick_volume": "volume", "real_volume": "volume"}
    df.rename(columns=rename, inplace=True)
    for col in ("open", "close", "high", "low"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' after normalise. Got: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df


def _save(df: pd.DataFrame, symbol: str, tf: str) -> Path:
    out_dir = DATA_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tf}.parquet"
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    return path


# ── MT5 downloader ────────────────────────────────────────────────────────────

def _mt5_download(
    symbol: str,
    tf:     str,
    start:  datetime,
    end:    datetime,
    verbose: bool = True,
) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError(
            "MetaTrader5 package not found. Install with:\n"
            "  uv add MetaTrader5\n"
            "or use --source yfinance"
        )

    server   = os.environ.get("MT5_SERVER", "")
    login    = int(os.environ.get("MT5_LOGIN", "0"))
    password = os.environ.get("MT5_PASSWORD", "")
    if not server or not login or not password:
        raise EnvironmentError(
            "MT5 credentials missing. Set MT5_SERVER, MT5_LOGIN, MT5_PASSWORD in .env"
        )
    if not mt5.initialize(server=server, login=login, password=password):
        raise RuntimeError(
            f"MT5 initialize() failed: {mt5.last_error()}\n"
            "Make sure the MT5 terminal is running and you are logged in."
        )

    tf_const = getattr(mt5, MT5_TF_MAP[tf])

    if verbose:
        print(f"  MT5 downloading {symbol} {tf}  {start.date()} -> {end.date()} …")

    rates = mt5.copy_rates_range(symbol, tf_const, start, end)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(
            f"MT5 returned no data for {symbol} {tf}. "
            f"Check terminal has the symbol and the date range is available."
        )

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("timestamp", inplace=True)
    df.drop(columns=["time", "spread", "real_volume"], errors="ignore", inplace=True)
    return _normalise_columns(df)


# ── OHLCV resampler ───────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a 1H DataFrame to a higher timeframe (e.g. '4H', '1D')."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


# ── yfinance downloader ───────────────────────────────────────────────────────

def _yf_download(
    symbol:  str,
    tf:      str,
    start:   datetime,
    end:     datetime,
    verbose: bool = True,
) -> pd.DataFrame:
    import yfinance as yf

    ticker = SYMBOL_MAP.get(symbol)
    if ticker is None:
        raise ValueError(f"No yfinance ticker mapping for '{symbol}'")

    # 4H: download 1H and resample
    fetch_tf  = tf
    resample_rule = None
    if tf == "4H":
        fetch_tf     = "1H"
        resample_rule = "4h"

    interval = YF_INTERVAL[fetch_tf]
    if verbose:
        label = tf if tf == fetch_tf else f"{tf} (via {fetch_tf} resample)"
        print(f"  yfinance downloading {symbol} ({ticker}) {label}  {start.date()} -> {end.date()} …")

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned empty data for {ticker} {tf}")

    df = _normalise_columns(raw)
    if resample_rule:
        df = _resample(df, resample_rule)
    return df


# ── Chunked yfinance (for long 1H histories) ──────────────────────────────────

def _yf_download_chunked(
    symbol:    str,
    tf:        str,
    start:     datetime,
    end:       datetime,
    chunk_days: int = 700,
    verbose:   bool = True,
) -> pd.DataFrame:
    """Download in chunks to bypass yfinance's per-request limits."""
    import yfinance as yf

    # 4H: fetch as 1H chunks and resample at the end
    fetch_tf      = "1H" if tf == "4H" else tf
    resample_rule = "4h" if tf == "4H" else None

    ticker   = SYMBOL_MAP[symbol]
    interval = YF_INTERVAL[fetch_tf]
    delta    = pd.Timedelta(days=chunk_days)
    chunks   = []
    cur      = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")

    while cur < end_ts:
        chunk_end = min(cur + delta, end_ts)
        if verbose:
            print(f"    chunk {cur.date()} -> {chunk_end.date()} …")
        raw = yf.download(
            ticker,
            start=cur.to_pydatetime(),
            end=chunk_end.to_pydatetime(),
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if not raw.empty:
            chunks.append(_normalise_columns(raw))
        cur = chunk_end + pd.Timedelta(seconds=1)

    if not chunks:
        raise ValueError(f"yfinance returned no data at all for {ticker} {tf}")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    if resample_rule:
        df = _resample(df, resample_rule)
    return df


# ── Per-symbol download ───────────────────────────────────────────────────────

def download_symbol(
    symbol:  str,
    source:  str,
    start:   datetime,
    end:     datetime,
    verbose: bool = True,
) -> dict[str, Path]:
    """Download all three timeframes for one symbol. Returns {tf: path}."""
    paths = {}
    for tf in TIMEFRAMES:
        try:
            if source == "mt5":
                df = _mt5_download(symbol, tf, start, end, verbose)
            else:
                # 1D: yfinance has full multi-year daily history, fetch directly
                # 4H: chunk 1H and resample (handles the 730-day limit)
                # 1H: chunk to handle 730-day limit
                # 15M/5M: max 60d, fetch directly
                if tf in ("1D",):
                    df = _yf_download(symbol, tf, start, end, verbose)
                elif tf in ("4H", "1H"):
                    df = _yf_download_chunked(symbol, tf, start, end, verbose=verbose)
                else:
                    df = _yf_download(symbol, tf, start, end, verbose)

            path = _save(df, symbol, tf)
            paths[tf] = path
            if verbose:
                print(f"    saved {len(df):>7,} bars -> {path.relative_to(DATA_DIR.parent)}")

        except Exception as exc:
            print(f"  [WARN] {symbol} {tf}: {exc}")
            paths[tf] = None

    return paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Download OHLCV data for the Hermes MTF backtest."
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Single symbol to download (e.g. EURUSD). Omit for all 10 pairs.",
    )
    parser.add_argument(
        "--source",
        choices=["mt5", "yfinance"],
        default="yfinance",
        help="Data source: mt5 (requires terminal) or yfinance (default fallback).",
    )
    parser.add_argument(
        "--start",
        default="2019-01-01",
        help="Start date  YYYY-MM-DD  (default: 2019-01-01).",
    )
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="End date  YYYY-MM-DD  (default: today).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the symbol map and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nSymbol map (MT5 -> yfinance):\n")
        for mt5_sym, yf_sym in SYMBOL_MAP.items():
            print(f"  {mt5_sym:<10s}  {yf_sym}")
        print()
        return

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    symbols = [args.symbol] if args.symbol else list(SYMBOL_MAP.keys())

    print(f"\nHermes Data Downloader")
    print(f"  source : {args.source}")
    print(f"  range  : {args.start} -> {args.end}")
    print(f"  symbols: {', '.join(symbols)}")
    print(f"  output : {DATA_DIR}\n")

    total_ok  = 0
    total_err = 0

    for sym in symbols:
        if sym not in SYMBOL_MAP:
            print(f"[ERROR] Unknown symbol '{sym}'. Run --list to see valid symbols.")
            continue
        print(f"[{sym}]")
        paths = download_symbol(sym, args.source, start_dt, end_dt, verbose=True)
        ok  = sum(1 for p in paths.values() if p is not None)
        err = sum(1 for p in paths.values() if p is None)
        total_ok  += ok
        total_err += err
        print()

    print(f"Done.  {total_ok} files saved,  {total_err} errors.")
    if total_err:
        print("  Re-run with --source mt5 (if terminal open) for higher-quality data.")


if __name__ == "__main__":
    main()
