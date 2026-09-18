"""Cleaning, feature building, and sequence creation.

Two feature sets are built and compared:
  base = the metal's own price and technical indicators
  full = base + external market drivers (dollar, yields, inflation, risk, oil, ...)

The model learns log returns, not raw prices, because returns stay in a
similar range over time. Predicted returns are turned back into prices.
"""
import numpy as np
import pandas as pd

import config
from backend.factors import factor_features

BASE_FEATURES = ["log_return", "ret_4", "ret_24", "ma_gap_24", "ma_gap_120",
                 "volatility", "rsi_14", "macd_hist", "hour_sin", "hour_cos"]
# a driver missing at some hours (for example Brent before 2007) is filled with its
# training average after scaling (scaled value 0), both in training and live.


def clean_prices(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[close > 0].dropna()
    if len(close) < 200:
        raise ValueError("Not enough price data to build features.")
    return close


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def base_features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": close})
    lp = np.log(df["close"])
    df["log_return"] = lp.diff()
    df["ret_4"] = lp.diff(4)                                   # momentum
    df["ret_24"] = lp.diff(24)
    df["ma_gap_24"] = df["close"] / df["close"].rolling(24).mean() - 1      # trend
    df["ma_gap_120"] = df["close"] / df["close"].rolling(120).mean() - 1
    df["volatility"] = df["log_return"].rolling(24).std()
    df["rsi_14"] = _rsi(df["close"]) / 100
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / df["close"]
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df


def build_features(close: pd.Series, metal: str, raw_factors: dict | None = None,
                   feature_set: str = "base", training: bool = False):
    """Returns (df, feature_columns).
    training=True : drop drivers that cover less than FEATURES_REQUIRED_SHARE of the rows.
    Rows are only dropped when the metal's own features are missing."""
    df = base_features(clean_prices(close))
    cols = list(BASE_FEATURES)
    if feature_set == "full" and raw_factors:
        ff = factor_features(raw_factors, df["close"], metal)
        if training:
            keep = ff.columns[ff.notna().mean() >= config.FEATURES_REQUIRED_SHARE]
            ff = ff[keep]
        df = df.join(ff)
        cols += list(ff.columns)
    df = df.dropna(subset=BASE_FEATURES)
    return df, cols


def make_sequences(features: np.ndarray, log_price: np.ndarray,
                   seq_len: int = config.SEQ_LENGTH, horizons=config.HORIZONS):
    """X[i]  = rows L-seq_len+1 .. L
    Y[i,k] = log_price[L + h_k] - log_price[L]   (return over the next h_k hours)"""
    max_h = max(horizons)
    X, Y, last_rows = [], [], []
    for L in range(seq_len - 1, len(features) - max_h):
        X.append(features[L - seq_len + 1:L + 1])
        Y.append([log_price[L + h] - log_price[L] for h in horizons])
        last_rows.append(L)
    return (np.array(X, dtype="float32"), np.array(Y, dtype="float32"),
            np.array(last_rows))
