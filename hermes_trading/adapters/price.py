import asyncio
import os

import numpy as np

SCHEMA_VERSION = "1.0"

# Exchanges tried in order until one succeeds.
# Kraken is first — no geo-restrictions, solid public OHLCV API.
# Binance is kept as last-resort for local dev where it's reachable.
_FALLBACK_EXCHANGES = ["kraken", "coinbaseprime", "bitstamp", "binance"]


class SchemaError(Exception):
    pass


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


async def _fetch_from_exchange(exchange_id: str, asset: str, api_key, api_secret) -> list:
    """Fetch OHLCV from one exchange, always closing the session cleanly."""
    import ccxt.async_support as ccxt

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls(
        {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
    )
    try:
        ohlcv = await exchange.fetch_ohlcv(asset, timeframe="1m", limit=50)
        return ohlcv
    finally:
        # Shield ensures close() runs even if the outer task is cancelled.
        try:
            await asyncio.shield(exchange.close())
        except Exception:
            pass


async def fetch(asset: str = "BTC/USDT") -> dict:
    api_key = os.getenv("EXCHANGE_API_KEY") or None
    api_secret = os.getenv("EXCHANGE_API_SECRET") or None

    # Respect explicit override; otherwise try fallback chain.
    forced = os.getenv("EXCHANGE_ID", "")
    exchange_order = [forced] if forced else _FALLBACK_EXCHANGES

    last_exc: Exception = SchemaError("No exchanges configured")
    for exchange_id in exchange_order:
        try:
            ohlcv = await _fetch_from_exchange(exchange_id, asset, api_key, api_secret)
            if ohlcv and len(ohlcv) >= 2:
                closes = [c[4] for c in ohlcv]
                last = ohlcv[-1]
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "asset": asset,
                    "exchange": exchange_id,
                    "price": float(last[4]),
                    "rsi": _compute_rsi(closes),
                    "volume": float(last[5]),
                    "high": float(last[2]),
                    "low": float(last[3]),
                }
                _validate(result)
                return result
            last_exc = SchemaError(f"{exchange_id}: insufficient OHLCV data")
        except SchemaError:
            raise
        except Exception as e:
            last_exc = e
            continue  # try next exchange

    raise last_exc


def _validate(data: dict):
    required = {"schema_version", "asset", "price", "rsi"}
    missing = required - data.keys()
    if missing:
        raise SchemaError(f"Price schema missing fields: {missing}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"Price schema version mismatch: {data['schema_version']} != {SCHEMA_VERSION}"
        )
