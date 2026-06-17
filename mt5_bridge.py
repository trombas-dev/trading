"""
mt5_bridge.py — MT5 -> PostgreSQL data bridge (Windows only).

Run this on your Windows PC alongside MT5 terminal.
It pushes OHLCV bars and live bid-ask spreads to Railway PostgreSQL
every 5 minutes so the Hermes loops on Railway get real MT5 data
instead of yfinance.

Setup:
  pip install MetaTrader5 psycopg2-binary python-dotenv schedule
  Set DATABASE_URL in .env (copy from Railway → Postgres → Connect)
  python mt5_bridge.py

DATABASE_URL format:
  postgresql://user:password@host:port/dbname
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta

import schedule
import psycopg2
import psycopg2.extras
import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge")

# ── Config ────────────────────────────────────────────────────────────────────

MT5_SERVER   = "ICMarketsEU-Demo"
MT5_LOGIN    = 52037890
MT5_PASSWORD = "$8lyQHf3PnjvAx"

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# All instruments used by either strategy
SYMBOLS = [
    # Fibonacci strategy
    "GBPAUD", "BTCUSD", "GBPUSD", "USDCHF", "XAUUSD", "NZDUSD",
    # Breakout strategy
    "US30", "US500", "JP225", "XTIUSD", "XBRUSD", "XNGUSD",
]

# Timeframes: MT5 constant name -> string label -> bar count to keep
TIMEFRAMES = {
    mt5.TIMEFRAME_H1:  ("1H",  2000),   # ~83 days
    mt5.TIMEFRAME_M15: ("15M",  600),   # ~6 days
    mt5.TIMEFRAME_M5:  ("5M",   600),   # ~2 days
}

PUSH_INTERVAL_S = 300   # every 5 minutes


# ── Database ──────────────────────────────────────────────────────────────────

def connect_db():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL not set. Add it to .env or set as environment variable.\n"
            "Copy it from Railway → your project → Postgres → Connect tab."
        )
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    conn.autocommit = False
    return conn


def ensure_schema(conn):
    """Create tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                symbol      VARCHAR(20)  NOT NULL,
                timeframe   VARCHAR(5)   NOT NULL,
                ts          TIMESTAMPTZ  NOT NULL,
                open        DOUBLE PRECISION,
                high        DOUBLE PRECISION,
                low         DOUBLE PRECISION,
                close       DOUBLE PRECISION,
                volume      DOUBLE PRECISION,
                PRIMARY KEY (symbol, timeframe, ts)
            );
            CREATE INDEX IF NOT EXISTS bars_sym_tf_ts
                ON bars (symbol, timeframe, ts DESC);

            CREATE TABLE IF NOT EXISTS spreads (
                symbol      VARCHAR(20)  PRIMARY KEY,
                spread      DOUBLE PRECISION,
                updated_at  TIMESTAMPTZ  NOT NULL
            );
        """)
    conn.commit()
    log.info("DB schema ready")


# ── MT5 ───────────────────────────────────────────────────────────────────────

def connect_mt5():
    if not mt5.initialize(server=MT5_SERVER, login=MT5_LOGIN, password=MT5_PASSWORD):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    info = mt5.account_info()
    log.info(f"MT5 connected: {info.login}  balance {info.balance:,.2f} {info.currency}")


def fetch_bars(symbol: str, tf_const, n_bars: int):
    """Fetch last n_bars from MT5 for symbol/timeframe."""
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
    if rates is None:
        return []
    if hasattr(rates, '__len__') and len(rates) == 0:
        return []
    return rates


def fetch_spread(symbol: str) -> float:
    """Return live bid-ask spread in price units."""
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 0.0
    return float(tick.ask - tick.bid)


# ── Push logic ────────────────────────────────────────────────────────────────

def push_bars(conn, symbol: str, tf_const, tf_label: str, n_bars: int) -> int:
    """Fetch bars from MT5 and upsert into DB. Returns rows inserted."""
    rates = fetch_bars(symbol, tf_const, n_bars)
    if rates is None or len(rates) == 0:
        return 0

    rows = [
        (
            symbol, tf_label,
            datetime.fromtimestamp(r["time"], tz=timezone.utc),
            float(r["open"]), float(r["high"]),
            float(r["low"]),  float(r["close"]),
            float(r["tick_volume"]),
        )
        for r in rates
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO bars (symbol, timeframe, ts, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, timeframe, ts) DO UPDATE
                SET open=EXCLUDED.open, high=EXCLUDED.high,
                    low=EXCLUDED.low,   close=EXCLUDED.close,
                    volume=EXCLUDED.volume
            """,
            rows,
            page_size=500,
        )
    conn.commit()
    return len(rows)


def push_spread(conn, symbol: str):
    spread = fetch_spread(symbol)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO spreads (symbol, spread, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE
                SET spread=EXCLUDED.spread, updated_at=EXCLUDED.updated_at
            """,
            (symbol, spread, datetime.now(timezone.utc)),
        )
    conn.commit()


def push_all_spreads(conn):
    """Push live spreads for all symbols — runs every minute."""
    for symbol in SYMBOLS:
        try:
            push_spread(conn, symbol)
        except Exception as exc:
            log.warning(f"spread {symbol}: {exc}")
            try:
                conn.rollback()
            except Exception:
                pass


# ── Main tick ─────────────────────────────────────────────────────────────────

def tick(conn, full: bool = False):
    """
    Push one cycle of bars + spreads.
    full=True on startup: push full history.
    full=False on interval: push only recent bars (last 10 per TF).
    """
    now = datetime.now(timezone.utc)
    total_bars = 0
    errors = 0

    for symbol in SYMBOLS:
        try:
            # Spread (always fresh)
            push_spread(conn, symbol)

            for tf_const, (tf_label, n_full) in TIMEFRAMES.items():
                n = n_full if full else 10
                inserted = push_bars(conn, symbol, tf_const, tf_label, n)
                total_bars += inserted

        except Exception as exc:
            log.warning(f"  {symbol}: {exc}")
            errors += 1
            try:
                conn.rollback()
            except Exception:
                pass

    log.info(
        f"[{'FULL' if full else 'TICK'}] "
        f"{now.strftime('%H:%M:%S')}  "
        f"bars={total_bars}  errors={errors}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not DATABASE_URL:
        print(
            "\nERROR: DATABASE_URL not set.\n"
            "  1. In Railway: add a Postgres plugin to your project\n"
            "  2. Go to Postgres → Connect → copy the DATABASE_URL\n"
            "  3. Paste it into your .env file here:\n"
            "       DATABASE_URL=postgresql://...\n"
        )
        sys.exit(1)

    log.info("MT5 Bridge starting...")
    connect_mt5()

    conn = connect_db()
    ensure_schema(conn)

    # Full history push on startup
    log.info(f"Pushing full history for {len(SYMBOLS)} symbols x {len(TIMEFRAMES)} timeframes...")
    tick(conn, full=True)
    log.info("Full history push complete. Starting 5-minute interval ticks.")

    # Spreads every minute, bars every 5 minutes
    schedule.every(1).minutes.do(push_all_spreads, conn=conn)
    schedule.every(5).minutes.do(tick, conn=conn, full=False)

    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    main()
