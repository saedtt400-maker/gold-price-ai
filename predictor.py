"""Loads the trained LSTM (base or full feature set) and makes predictions."""
import joblib
import numpy as np
import pandas as pd

import config
from backend.processing import build_features


class PredictionError(Exception):
    pass


class MetalPredictor:
    def __init__(self, metal: str):
        from tensorflow import keras
        self.metal = metal
        mp, sp = config.model_path(metal), config.scaler_path(metal)
        if not mp.exists() or not sp.exists():
            raise PredictionError(f"Model for {metal} not found. Run: python -m model.train_lstm")
        self.model = keras.models.load_model(mp)
        saved = joblib.load(sp)
        self.x_scaler = saved["x_scaler"]
        self.y_std = np.asarray(saved["y_std"])
        self.horizons = list(saved["horizons"])
        self.features = list(saved["features"])
        self.feature_set = saved.get("feature_set", "base")
        self.interval_q = saved.get("interval_q", {})
        self.missing = []
        if self.horizons != list(config.HORIZONS):
            raise PredictionError("Saved model horizons differ from config. Please retrain.")

    def _matrix(self, close: pd.Series, raw: dict):
        df, _ = build_features(close, self.metal, raw, self.feature_set)
        if len(df) < config.SEQ_LENGTH:
            raise PredictionError("Not enough recent data to make a prediction.")
        X = df.reindex(columns=self.features)
        # a driver that is down right now -> use its training average (scaled value 0)
        means = pd.Series(self.x_scaler.mean_, index=self.features)
        self.missing = [c for c in self.features if X[c].tail(config.SEQ_LENGTH).isna().any()]
        X = X.fillna(means)
        return df, self.x_scaler.transform(X)

    def predict(self, close: pd.Series, raw: dict | None = None) -> dict:
        """{hours: {"price": p, "bands": {"0.95": [low, high], ...}}}"""
        try:
            df, feats = self._matrix(close, raw or {})
            window = feats[-config.SEQ_LENGTH:]
            rets = self.model.predict(window[np.newaxis], verbose=0)[0] * self.y_std
            last = float(df["close"].iloc[-1])
            out = {}
            for k, (h, r) in enumerate(zip(self.horizons, rets)):
                p = float(last * np.exp(r))
                bands = {c: [p * float(np.exp(-q[k])), p * float(np.exp(q[k]))]
                         for c, q in self.interval_q.items()}
                out[h] = {"price": p, "bands": bands}
            return out
        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}")

    def rolling_predictions(self, close: pd.Series, raw: dict | None = None,
                            hours: int = 72) -> pd.DataFrame:
        try:
            df, feats = self._matrix(close, raw or {})
            n, s = len(df), config.SEQ_LENGTH
            L = np.arange(max(s - 1, n - 1 - hours), n - 1)
            if len(L) == 0:
                raise PredictionError("Not enough data for the comparison chart.")
            X = np.stack([feats[i - s + 1:i + 1] for i in L])
            ret1 = self.model.predict(X, verbose=0)[:, 0] * self.y_std[0]
            return pd.DataFrame({"time": df.index[L + 1],
                                 "actual": df["close"].values[L + 1],
                                 "predicted": df["close"].values[L] * np.exp(ret1)})
        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Comparison failed: {e}")
