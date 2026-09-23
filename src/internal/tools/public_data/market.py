"""Market-data tools: stock quotes, crypto prices, currency conversion.

All three answer with a flat JSON object of facts rather than documents, so
none of them is citeable.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ..base import FunctionTool, ToolEffect
from ._http import PublicDataError, get_json, guarded

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
EXCHANGE_RATE_URL = "https://api.exchangerate-api.com/v4/latest/{base}"

# Yahoo 4xxs anything that does not look like a browser.
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}

# _SYMBOL_RE's match is interpolated into a URL path, so it is validated
# rather than escaped — a stray "../" would otherwise retarget the request.
# _CURRENCY_RE guards both currency codes: `from_currency` is interpolated into
# the URL path the same way, while `to_currency` never reaches a URL — it is
# used only as a dict key against the response's `rates`, where the same shape
# check keeps a malformed code from failing silently as a missing-key lookup.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9.^=-]{1,15}$")
_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")

# CoinGecko keys on slugs, not tickers; models say "btc".
_COIN_IDS = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "usdt": "tether",
    "bnb": "binancecoin",
    "sol": "solana",
    "xrp": "ripple",
    "usdc": "usd-coin",
    "ada": "cardano",
    "doge": "dogecoin",
    "trx": "tron",
    "dot": "polkadot",
    "matic": "matic-network",
    "dai": "dai",
    "shib": "shiba-inu",
    "avax": "avalanche-2",
}


# ------------------------------------------------------------------- stocks

_STOCK_PARAMS = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "minLength": 1,
            "description": "Ticker symbol, e.g. AAPL or TSLA.",
        },
    },
    "required": ["symbol"],
    "additionalProperties": False,
}


async def _get_stock_quote(symbol: str) -> dict:
    symbol = symbol.strip()
    if not _SYMBOL_RE.match(symbol):
        raise PublicDataError(f"invalid ticker symbol {symbol!r}")

    payload = await get_json(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={"interval": "1d", "range": "1d"},
        headers=_YAHOO_HEADERS,
    )
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        raise PublicDataError(f"no quote data for symbol {symbol!r}")

    meta = results[0].get("meta") or {}
    return {
        "symbol": symbol.upper(),
        "currency": meta.get("currency", "USD"),
        "current_price": meta.get("regularMarketPrice"),
        "previous_close": meta.get("previousClose") or meta.get("chartPreviousClose"),
        "day_high": meta.get("regularMarketDayHigh"),
        "day_low": meta.get("regularMarketDayLow"),
        "volume": meta.get("regularMarketVolume"),
        "exchange": meta.get("exchangeName"),
    }


def build_stock_quote_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_get_stock_quote),
        name="get_stock_quote",
        description=(
            "Get the latest price, day range, and volume for a listed stock or "
            "ETF by ticker symbol."
        ),
        parameters=_STOCK_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


# ------------------------------------------------------------------- crypto

_CRYPTO_PARAMS = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "minLength": 1,
            "description": "Coin ticker or CoinGecko id, e.g. btc or bitcoin.",
        },
        "vs_currency": {
            "type": "string",
            "pattern": "^[A-Za-z]{2,10}$",
            "description": "Currency to price in, e.g. usd or eur.",
            "default": "usd",
        },
    },
    "required": ["symbol"],
    "additionalProperties": False,
}


async def _get_crypto_price(symbol: str, vs_currency: str = "usd") -> dict:
    key = symbol.strip().lower()
    coin_id = _COIN_IDS.get(key, key)
    vs = vs_currency.strip().lower()

    payload = await get_json(
        COINGECKO_PRICE_URL,
        params={
            "ids": coin_id,
            "vs_currencies": vs,
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        },
    )
    data = (payload or {}).get(coin_id)
    if not data:
        raise PublicDataError(f"unknown cryptocurrency {symbol!r}")

    updated = data.get("last_updated_at")
    return {
        "symbol": symbol.upper(),
        "coin_id": coin_id,
        "currency": vs.upper(),
        "price": data.get(vs),
        "market_cap": data.get(f"{vs}_market_cap"),
        "volume_24h": data.get(f"{vs}_24h_vol"),
        "change_24h_percent": data.get(f"{vs}_24h_change"),
        "last_updated": (
            datetime.fromtimestamp(updated, tz=timezone.utc).isoformat()
            if updated
            else None
        ),
    }


def build_crypto_price_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_get_crypto_price),
        name="get_crypto_price",
        description=(
            "Get the current price, market cap, and 24-hour change for a "
            "cryptocurrency such as bitcoin or ethereum."
        ),
        parameters=_CRYPTO_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


# ----------------------------------------------------------------- currency

_CURRENCY_PARAMS = {
    "type": "object",
    "properties": {
        "amount": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "How much to convert.",
        },
        "from_currency": {
            "type": "string",
            "pattern": "^[A-Za-z]{3}$",
            "minLength": 3,
            "description": "Three-letter source currency code, e.g. USD.",
        },
        "to_currency": {
            "type": "string",
            "pattern": "^[A-Za-z]{3}$",
            "minLength": 3,
            "description": "Three-letter target currency code, e.g. EUR.",
        },
    },
    "required": ["amount", "from_currency", "to_currency"],
    "additionalProperties": False,
}


def _currency_code(value: str, label: str) -> str:
    code = value.strip()
    if not _CURRENCY_RE.match(code):
        raise PublicDataError(
            f"{label} must be a 3-letter currency code, got {value!r}"
        )
    return code.upper()


async def _convert_currency(
    amount: float, from_currency: str, to_currency: str
) -> dict:
    base = _currency_code(from_currency, "from_currency")
    target = _currency_code(to_currency, "to_currency")

    payload = await get_json(EXCHANGE_RATE_URL.format(base=base))
    rates = payload.get("rates") or {}
    if target not in rates:
        raise PublicDataError(f"no exchange rate from {base} to {target}")

    rate = rates[target]
    return {
        "amount": float(amount),
        "from_currency": base,
        "to_currency": target,
        "rate": rate,
        "converted_amount": round(float(amount) * rate, 6),
        "date": payload.get("date"),
    }


def build_currency_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_convert_currency),
        name="convert_currency",
        description=(
            "Convert an amount of money from one currency to another at "
            "today's exchange rate."
        ),
        parameters=_CURRENCY_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )
