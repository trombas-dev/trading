import os
import time

import httpx

SCHEMA_VERSION = "1.0"
_CACHE_TTL_S = 300  # 5-minute cache — CoinGecko free tier is rate-limited per IP

_cache: dict = {"ts": 0.0, "data": None}


class SchemaError(Exception):
    pass


async def fetch() -> dict:
    now = time.monotonic()
    if _cache["data"] is not None and (now - _cache["ts"]) < _CACHE_TTL_S:
        return _cache["data"]

    api_key = os.getenv("GLASSNODE_API_KEY", "")
    result = await (_fetch_glassnode(api_key) if api_key else _fetch_free())

    _cache["ts"] = now
    _cache["data"] = result
    return result


async def _fetch_glassnode(api_key: str) -> dict:
    url = "https://api.glassnode.com/v1/metrics/indicators/sopr"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            url,
            params={"a": "BTC", "api_key": api_key, "i": "24h", "f": "JSON"},
        )
        resp.raise_for_status()
        data = resp.json()

    sopr = data[-1]["v"] if data else None
    result = {"schema_version": SCHEMA_VERSION, "source": "glassnode", "sopr": sopr}
    _validate(result)
    return result


async def _fetch_free() -> dict:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("bitcoin", {})

    result = {
        "schema_version": SCHEMA_VERSION,
        "source": "coingecko_free",
        "market_cap_usd": data.get("usd_market_cap"),
        "volume_24h_usd": data.get("usd_24h_vol"),
        "price_change_24h_pct": data.get("usd_24h_change"),
    }
    _validate(result)
    return result


def _validate(data: dict):
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"Onchain schema version mismatch: {data.get('schema_version')} != {SCHEMA_VERSION}"
        )
