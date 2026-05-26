import os
import time

import httpx

SCHEMA_VERSION = "1.0"
_CACHE_TTL_S = 300  # 5-minute cache

_cache: dict = {"ts": 0.0, "data": None}


class SchemaError(Exception):
    pass


async def fetch() -> dict:
    now = time.monotonic()
    if _cache["data"] is not None and (now - _cache["ts"]) < _CACHE_TTL_S:
        return _cache["data"]

    api_key = os.getenv("NEWS_API_KEY", "")
    result = await (_fetch_newsapi(api_key) if api_key else _fetch_free())

    _cache["ts"] = now
    _cache["data"] = result
    return result


async def _fetch_newsapi(api_key: str) -> dict:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "bitcoin OR crypto",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": api_key,
        "language": "en",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

    result = {
        "schema_version": SCHEMA_VERSION,
        "source": "newsapi",
        "article_count": len(articles),
        "sentiment_score": _simple_sentiment(articles),
    }
    _validate(result)
    return result


async def _fetch_free() -> dict:
    """
    CryptoPanic public endpoint is unreliable — fall back to a neutral
    sentiment placeholder so the loop is never blocked by news data.
    """
    result = {
        "schema_version": SCHEMA_VERSION,
        "source": "none",
        "article_count": 0,
        "sentiment_score": 0.0,
    }

    try:
        # Try Alternative.me Fear & Greed Index — stable free public API.
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json().get("data", [{}])[0]

        value = int(data.get("value", 50))
        # Map 0–100 Fear/Greed to -1..+1 sentiment
        sentiment = round((value - 50) / 50, 3)
        result.update({
            "source": "fear_greed",
            "fear_greed_value": value,
            "fear_greed_label": data.get("value_classification", ""),
            "sentiment_score": sentiment,
        })
    except Exception:
        pass  # neutral fallback stays in place

    _validate(result)
    return result


def _simple_sentiment(articles: list[dict]) -> float:
    positive = {"bull", "surge", "rally", "gain", "high", "rise", "up", "buy"}
    negative = {"bear", "crash", "drop", "fall", "low", "down", "sell", "fear"}
    scores = []
    for a in articles:
        words = set((a.get("title", "") + " " + a.get("description", "")).lower().split())
        pos = len(words & positive)
        neg = len(words & negative)
        if pos + neg > 0:
            scores.append((pos - neg) / (pos + neg))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _validate(data: dict):
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"News schema version mismatch: {data.get('schema_version')} != {SCHEMA_VERSION}"
        )
