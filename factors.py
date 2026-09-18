"""External market drivers: download, place at their public time, build features.

Slow data (daily, weekly, monthly) is turned into features at its own frequency
first (for example a 5-day change), then placed at the moment it became public
and carried forward hour by hour. The model never sees data from the future.
"""
import io
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config
from backend.history_sources import dxy_hourly, hourly_close, yahoo_daily
from backend.price_client import ExternalAPIError

log = logging.getLogger("factors")


# ---------- download helpers ----------
def fetch_fred(series_id: str) -> pd.Series:
    r = requests.get(config.FRED_CSV_URL, params={"id": series_id, "cosd": config.FRED_START},
                     timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0])).dropna()
    if s.empty:
        raise ExternalAPIError(f"FRED returned no data for {series_id}")
    return s


def fetch_cot(metal: str) -> pd.Series:
    market = config.COT_MARKETS[metal]
    params = {"$where": f"market_and_exchange_names = '{market}'",
              "$order": "report_date_as_yyyy_mm_dd", "$limit": 5000}
    r = requests.get(config.CFTC_COT_URL, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise ExternalAPIError(f"CFTC returned no COT data for {metal}")
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    net = (num("noncomm_positions_long_all") - num("noncomm_positions_short_all")) / num("open_interest_all")
    s = pd.Series(net.values, index=pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None))
    s = s.dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_gpr() -> pd.Series:
    r = requests.get(config.GPR_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content))
    date_col = next(c for c in df.columns if str(c).lower() in ("date", "day"))
    s = pd.Series(pd.to_numeric(df["GPRD"], errors="coerce").values,
                  index=pd.to_datetime(df[date_col])).dropna()
    return s.sort_index()


def _public(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Date index -> UTC time when the value became public (next midnight + lag)."""
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx.normalize() + pd.Timedelta(days=lag_days + 1)
    return out.sort_index()


# ---------- native-frequency transforms ----------
def slow_features(key: str, spec: dict, s: pd.Series) -> pd.DataFrame:
    kind = spec["kind"]
    s = s.astype(float)
    if kind == "daily_price":
        lv = np.log(s[s > 0])
        df = pd.DataFrame({"ret1d": lv.diff(), "ret5d": lv.diff(5)})
    elif kind == "daily_rate":
        df = pd.DataFrame({"lvl": s, "d5": s.diff(5)})
    elif kind == "daily_volume":
        lv = np.log1p(s)
        df = pd.DataFrame({"z60": (lv - lv.rolling(60).mean()) / lv.rolling(60).std()})
    elif kind == "yoy":
        df = pd.DataFrame({"lvl": s.pct_change(12, fill_method=None) * 100})
    elif kind == "loglevel":
        df = pd.DataFrame({"lvl": np.log(s[s > 0])})
    elif kind == "weekly_level":
        df = pd.DataFrame({"lvl": s, "d4w": s.diff(4)})
    else:
        raise ValueError(f"Unknown kind {kind}")
    df.columns = [f"{key}__{c}" for c in df.columns]
    return _public(df.replace([np.inf, -np.inf], np.nan).dropna(how="all"), spec["lag_days"])


def specs_for(metal: str) -> dict:
    return {k: v for k, v in config.FACTORS.items() if metal in v["metals"]}


class FactorStore:
    """Raw driver data for one metal.
    raw[key] = {"hourly": Series} or {"slow": DataFrame of ready features}
    display[key] = Series of the plain value (for the dashboard)."""

    def __init__(self, metal: str):
        self.metal = metal
        self.raw, self.display, self.status = {}, {}, {}
        self.last = {"hourly": None, "daily": None, "slow": None}

    def refresh(self, start_hourly: str, start_daily: str, groups=("hourly", "daily", "slow")):
        for key, spec in specs_for(self.metal).items():
            src = spec["source"]
            group = "hourly" if src in ("dk_fx", "dxy_synth") else "daily" if src == "yahoo_d" else "slow"
            if group not in groups:
                continue
            try:
                if src == "dxy_synth":
                    s = dxy_hourly(start_hourly)
                    self.raw[key] = {"hourly": s}
                elif src == "dk_fx":
                    s = hourly_close(spec["id"], start_hourly)
                    self.raw[key] = {"hourly": s}
                else:
                    if src == "yahoo_d":
                        field = "Volume" if spec["kind"] == "daily_volume" else "Close"
                        s = yahoo_daily(spec["id"], start_daily, field)
                    elif src == "fred":
                        s = fetch_fred(spec["id"])
                    elif src == "cot":
                        s = fetch_cot(self.metal)
                    elif src == "gpr":
                        s = fetch_gpr()
                    else:
                        raise ValueError(src)
                    self.raw[key] = {"slow": slow_features(key, spec, s)}
                    if spec["kind"] == "yoy":
                        s = s.pct_change(12, fill_method=None) * 100
                self.display[key] = s.dropna()
                self.status[key] = "ok"
            except Exception as e:
                self.status[key] = f"unavailable: {e}"
                log.warning("%s / %s: %s", self.metal, key, e)
        now = datetime.now(timezone.utc)
        for g in groups:
            self.last[g] = now

    def due(self, group: str, seconds: int) -> bool:
        t = self.last[group]
        return t is None or (datetime.now(timezone.utc) - t).total_seconds() > seconds

    def latest(self) -> dict:
        out = {}
        for key, s in self.display.items():
            if s.empty:
                continue
            spec = config.FACTORS[key]
            last = float(s.iloc[-1])
            ch = None
            if spec["source"] in ("dk_fx", "dxy_synth"):
                prev = s[s.index <= s.index[-1] - pd.Timedelta(hours=24)]
                if not prev.empty:
                    ch = (last / float(prev.iloc[-1]) - 1) * 100
            elif len(s) > 1 and spec["kind"] in ("daily_price",):
                ch = (last / float(s.iloc[-2]) - 1) * 100
            elif len(s) > 1 and spec["kind"] == "daily_rate":
                ch = last - float(s.iloc[-2])
            out[key] = {"value": last, "change": ch, "as_of": str(s.index[-1])}
        return out


# ---------- align to the hourly price ----------
def _asof(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    def to_ns(values):
        return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")

    left = pd.DataFrame({"t": to_ns(index)})
    right = frame.reset_index()
    right = right.rename(columns={right.columns[0]: "t"})
    right["t"] = to_ns(right["t"])
    right = right.sort_values("t").dropna(subset=["t"])
    m = pd.merge_asof(left, right, on="t", direction="backward")
    return m.drop(columns="t").set_index(index)


def factor_features(raw: dict, close: pd.Series, metal: str) -> pd.DataFrame:
    idx = close.index
    parts = []
    for key, d in raw.items():
        spec = config.FACTORS.get(key)
        if spec is None or metal not in spec["metals"]:
            continue
        if "hourly" in d:
            v = _asof(d["hourly"].rename("v").to_frame(), idx)["v"]
            lv = np.log(v)
            parts.append(pd.DataFrame({f"{key}__ret1": lv.diff(), f"{key}__ret24": lv.diff(24)},
                                      index=idx))
        else:
            parts.append(_asof(d["slow"], idx))
    if not parts:
        return pd.DataFrame(index=idx)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def group_of(feature: str) -> str:
    return feature.split("__")[0] if "__" in feature else "own_price"
