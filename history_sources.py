"""Long free history.

Dukascopy (spot prices, hourly candles, around 2003 to now) is the main source.
The newest hours that Dukascopy has not published yet are filled from Yahoo,
scaled to match Dukascopy on the overlapping hours.
Downloaded months are cached in data/cache so the full download happens once.
"""
import io
import logging
import lzma
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import config
from backend.price_client import ExternalAPIError, yahoo_hourly

log = logging.getLogger("history")
RECENT_TTL_SECONDS = 15 * 60
REC = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"),
                ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (student project)"


# ---------- Dukascopy ----------
def _get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 404:
                return b""
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            if i == tries - 1:
                raise ExternalAPIError(f"Dukascopy download failed: {url} ({e})")
            time.sleep(2 ** i)
    return b""


def _decode(content: bytes, base: pd.Timestamp, sym: str) -> pd.DataFrame:
    if not content:
        return pd.DataFrame()
    raw = lzma.decompress(content)
    arr = np.frombuffer(raw[: len(raw) // 24 * 24], dtype=REC)
    if arr.size == 0:
        return pd.DataFrame()
    df = pd.DataFrame({
        "open": arr["o"].astype(float), "high": arr["h"].astype(float),
        "low": arr["l"].astype(float), "close": arr["c"].astype(float),
        "volume": arr["v"].astype(float),
    }, index=base + pd.to_timedelta(arr["t"].astype("int64"), unit="s"))
    df = df[df["volume"] > 0]                       # drop flat weekend / holiday candles
    return _scale(df, sym)


def _scale(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Divide by the known scale; if the result looks wrong, try other scales."""
    if df.empty:
        return df
    spec = config.DUKASCOPY_SYMBOLS[sym]
    lo, hi = spec["range"]
    med = float(df["close"].median())
    for d in [spec["divisor"], 1000, 100000, 100, 10, 1, 10000]:
        if lo <= med / d <= hi:
            df[["open", "high", "low", "close"]] /= d
            return df
    raise ExternalAPIError(f"Could not find the price scale for {sym} (median raw {med})")


def _month_from_minutes(sym: str, y: int, m: int, after=None) -> pd.DataFrame:
    """Build hourly candles from daily 1-minute files, only for days after `after`."""
    today = datetime.now(timezone.utc).date()
    parts = []
    for d in pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D"):
        if d.month != m or d.date() > today:
            break
        if after is not None and d.date() < after.date():
            continue
        url = config.DUKASCOPY_MIN_URL.format(sym=sym, y=y, m0=m - 1, d=d.day)
        part = _decode(_get(url), pd.Timestamp(d, tz="UTC"), sym)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    mins = pd.concat(parts)
    return mins.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()


def _month(sym: str, y: int, m: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    is_recent = (y, m) >= ((now - pd.Timedelta(days=40)).year, (now - pd.Timedelta(days=40)).month)
    cache = config.CACHE_DIR / "dukascopy" / sym / f"{y}-{m:02d}.csv"
    if cache.exists():
        age = time.time() - cache.stat().st_mtime
        if not is_recent or age < RECENT_TTL_SECONDS:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df

    url = config.DUKASCOPY_URL.format(sym=sym, y=y, m0=m - 1)
    df = _decode(_get(url), pd.Timestamp(year=y, month=m, day=1, tz="UTC"), sym)
    if is_recent:
        # the hourly file of a running month lags; add the newest days from minute files
        after = df.index.max() if not df.empty else None
        mins = _month_from_minutes(sym, y, m, after)
        if not mins.empty:
            df = mins if df.empty else pd.concat([df, mins[mins.index > df.index.max()]])
    if not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df


def dukascopy_hourly(sym: str, start: str) -> pd.DataFrame:
    """Hourly OHLCV, index = bar CLOSE time (UTC)."""
    now = pd.Timestamp.now(tz="UTC")
    start_ts = pd.Timestamp(start)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    months = [(d.year, d.month) for d in pd.date_range(start_ts, now, freq="MS")]
    if not months or months[-1] != (now.year, now.month):
        months.append((now.year, now.month))
    with ThreadPoolExecutor(config.DUKASCOPY_WORKERS) as ex:
        parts = list(ex.map(lambda ym: _month(sym, *ym), months))
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise ExternalAPIError(f"No Dukascopy data for {sym}")
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = df.index + pd.Timedelta(hours=1)
    return df[df.index >= pd.Timestamp(start, tz="UTC")]


def hourly_close(sym: str, start: str) -> pd.Series:
    """Dukascopy close + Yahoo for the newest hours (scaled on the overlap)."""
    dk = None
    try:
        dk = dukascopy_hourly(sym, start)["close"]
    except ExternalAPIError as e:
        log.warning("%s", e)
    ticker = config.YAHOO_FX.get(sym)
    try:
        yh = yahoo_hourly(ticker, "60d") if ticker else None
    except ExternalAPIError as e:
        log.warning("%s", e)
        yh = None

    if dk is None and yh is None:
        raise ExternalAPIError(f"No hourly data for {sym} from any source")
    if dk is None:
        return yh
    if yh is None or yh.empty:
        return dk
    both = pd.concat([dk, yh], axis=1, join="inner").dropna()
    ratio = float((both.iloc[:, 0] / both.iloc[:, 1]).median()) if len(both) >= 10 \
        else float(dk.iloc[-1] / yh[yh.index <= dk.index[-1]].iloc[-1]) \
        if (yh.index <= dk.index[-1]).any() else 1.0
    tail = yh[yh.index > dk.index[-1]] * ratio
    out = pd.concat([dk, tail])
    out.name = sym
    return out


def dxy_hourly(start: str) -> pd.Series:
    """US Dollar Index rebuilt from its 6 currency pairs."""
    parts = {s: hourly_close(s, start) for s in config.DXY_WEIGHTS}
    df = pd.concat(parts, axis=1).sort_index().ffill().dropna()
    logv = np.log(config.DXY_CONST) + sum(w * np.log(df[s]) for s, w in config.DXY_WEIGHTS.items())
    return pd.Series(np.exp(logv), index=df.index, name="DXY")


# ---------- Yahoo daily ----------
def yahoo_daily(ticker: str, start: str, field: str = "Close") -> pd.Series:
    try:
        df = yf.download(ticker, start=start, interval="1d", progress=False, auto_adjust=False)
    except Exception as e:
        raise ExternalAPIError(f"Yahoo daily request failed for {ticker}: {e}")
    if df is None or df.empty or field not in df:
        raise ExternalAPIError(f"No daily {field} for {ticker}")
    col = df[field]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    col = pd.to_numeric(col, errors="coerce").dropna()
    col.index = pd.to_datetime(col.index).tz_localize(None)
    return col[~col.index.duplicated(keep="last")].sort_index()


def metal_hourly(metal: str, start: str) -> pd.Series:
    """Hourly spot close of a metal (USD/oz), index = bar close time."""
    s = hourly_close(config.ALL_METALS[metal]["dukascopy"], start)
    s = s[s > 0]
    s.name = "close"
    return s
