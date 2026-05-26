import asyncio

SCHEMA_VERSION = "1.0"


class SchemaError(Exception):
    pass


async def fetch() -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_sync)


def _fetch_sync() -> dict:
    dxy_price = None
    dxy_change = None
    spy_change = None

    try:
        import yfinance as yf

        hist = yf.Ticker("DX-Y.NYB").history(period="2d")
        if not hist.empty:
            dxy_price = round(float(hist["Close"].iloc[-1]), 4)
        if len(hist) >= 2:
            dxy_change = round(
                float(
                    (hist["Close"].iloc[-1] - hist["Close"].iloc[-2])
                    / hist["Close"].iloc[-2]
                    * 100
                ),
                4,
            )
    except Exception:
        pass

    try:
        import yfinance as yf

        hist = yf.Ticker("SPY").history(period="2d")
        if len(hist) >= 2:
            spy_change = round(
                float(
                    (hist["Close"].iloc[-1] - hist["Close"].iloc[-2])
                    / hist["Close"].iloc[-2]
                    * 100
                ),
                4,
            )
    except Exception:
        pass

    result = {
        "schema_version": SCHEMA_VERSION,
        "dxy_price": dxy_price,
        "dxy_change_pct": dxy_change,
        "spy_change_pct": spy_change,
    }
    _validate(result)
    return result


def _validate(data: dict):
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"Macro schema version mismatch: {data.get('schema_version')} != {SCHEMA_VERSION}"
        )
