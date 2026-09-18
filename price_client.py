"""Talks to gold-api.com (live spot price) and Yahoo Finance (hourly helper, USD/ILS)."""
import pandas as pd
import requests
import yfinance as yf

import config


class ExternalAPIError(Exception):
    """Raised when an external API fails or returns bad data."""


def fetch_live_price(metal: str) -> dict:
    """Latest spot price in USD per troy ounce from gold-api.com."""
    symbol = config.METALS[metal]["symbol"]
    url = config.LIVE_PRICE_URL.format(symbol=symbol)
    try:
        r = requests.get(url, timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        raise ExternalAPIError("The price API did not answer in time.")
    except requests.exceptions.ConnectionError:
        raise ExternalAPIError("No connection to the price API.")
    except requests.exceptions.HTTPError as e:
        raise ExternalAPIError(f"The price API returned an error: {e}")
    except ValueError:
        raise ExternalAPIError("The price API returned invalid JSON.")

    price = data.get("price")
    if price is None:
        raise ExternalAPIError("The price is missing in the API response.")
    try:
        price = float(price)
    except (TypeError, ValueError):
        raise ExternalAPIError("The price in the API response is not a number.")
    if price <= 0:
        raise ExternalAPIError("The API returned a price that is not valid.")

    return {
        "metal": metal,
        "symbol": symbol,
        "price_usd_oz": price,
        "api_updated_at": data.get("updatedAt"),
    }


def yahoo_hourly(ticker: str, period: str, field: str = "Close") -> pd.Series:
    """Hourly series from Yahoo Finance.
    The index is moved to the END of each bar (start + 1 hour), which is the moment
    the value is really known. This avoids using future information."""
    try:
        df = yf.download(ticker, period=period, interval="1h",
                         progress=False, auto_adjust=False)
    except Exception as e:
        raise ExternalAPIError(f"Yahoo Finance request failed for {ticker}: {e}")
    if df is None or df.empty or field not in df:
        raise ExternalAPIError(f"No {field} data returned for {ticker}.")
    col = df[field]
    if isinstance(col, pd.DataFrame):          # newer yfinance returns 2D columns
        col = col.iloc[:, 0]
    col = pd.to_numeric(col, errors="coerce").dropna()
    idx = pd.to_datetime(col.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    col.index = idx + pd.Timedelta(hours=1)
    col = col[~col.index.duplicated(keep="last")].sort_index()
    col.name = ticker
    return col


def fetch_usd_ils() -> float:
    """How many Israeli shekels for 1 US dollar."""
    try:
        df = yf.download(config.USD_ILS_TICKER, period="5d", interval="1h",
                         progress=False, auto_adjust=False)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rate = float(close.dropna().iloc[-1])
    except Exception as e:
        raise ExternalAPIError(f"Could not get the USD/ILS rate: {e}")
    if rate <= 0:
        raise ExternalAPIError("USD/ILS rate is not valid.")
    return rate
