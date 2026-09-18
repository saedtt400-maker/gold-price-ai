"""Train the gold LSTM on the full free history (hourly since about 2003,
daily drivers since 2000) and compare two feature sets:

  base = gold's own price + technical indicators
  full = base + market drivers (dollar, yields, inflation, risk, oil, stocks, ETF, COT)

Outputs per horizon (1, 4, 8, 12, 24 hours):
  MAE, RMSE, MAPE, price accuracy (100 - MAPE), R2, direction accuracy,
  naive baseline, and confidence bands (80/95/99%) with their real test coverage.

Run from the project folder:
    python -m model.train_lstm
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import config
from backend.factors import FactorStore, group_of
from backend.history_sources import metal_hourly
from backend.processing import build_features, clean_prices

H = config.HORIZONS
S = config.SEQ_LENGTH
MAXH = max(H)


# ---------- model ----------
def build_model(n_features: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(S, n_features)),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(len(H)),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=1.0))
    return model


def price_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "MAPE_%": mape, "price_accuracy_%": 100 - mape,
            "R2": float(r2_score(actual, predicted))}


# ---------- data ----------
def prepare(close, raw, fs, cut_train, cut_val):
    df, cols = build_features(close, "gold", raw, fs, training=True)
    t = df.index
    scaler = StandardScaler().fit(df.loc[t < cut_train, cols])     # NaN ignored in fit
    feats = np.nan_to_num(scaler.transform(df[cols]), nan=0.0).astype("float32")
    lp = np.log(df["close"].values)
    n = len(df)
    Y = np.full((n, len(H)), np.nan, dtype="float32")
    for k, h in enumerate(H):
        Y[: n - h, k] = lp[h:] - lp[: n - h]
    L = np.arange(S - 1, n - MAXH)
    tL, tEnd = t[L], t[L + MAXH]
    split = {"tr": L[tEnd < cut_train],
             "va": L[(tL >= cut_train) & (tEnd < cut_val)],
             "te": L[tL >= cut_val]}
    return dict(df=df, cols=cols, scaler=scaler, feats=feats, Y=Y, split=split)


def make_ds(d, L, y_std, shuffle=False):
    feats = tf.constant(d["feats"])
    Ys = tf.constant(np.nan_to_num(d["Y"] / y_std))
    ds = tf.data.Dataset.from_tensor_slices(L.astype("int32"))
    if shuffle:
        ds = ds.shuffle(len(L), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(lambda i: (tf.gather(feats, tf.range(i - S + 1, i + 1)), Ys[i]),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def windows(d, L):
    return np.stack([d["feats"][i - S + 1:i + 1] for i in L])


# ---------- train / evaluate ----------
def train_set(d):
    y_std = np.nanstd(d["Y"][d["split"]["tr"]], axis=0)
    keras.utils.set_random_seed(42)
    model = build_model(len(d["cols"]))
    hist = model.fit(
        make_ds(d, d["split"]["tr"], y_std, shuffle=True),
        validation_data=make_ds(d, d["split"]["va"], y_std),
        epochs=config.EPOCHS, verbose=2, shuffle=False,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])
    return model, y_std, float(min(hist.history["val_loss"]))


def predict_returns(model, d, L, y_std, X=None):
    src = make_ds(d, L, y_std) if X is None else X
    return model.predict(src, verbose=0, batch_size=config.BATCH_SIZE) * y_std


def interval_quantiles(model, d, y_std):
    """Size of the band that contains 80/95/99% of validation errors (per horizon)."""
    L = d["split"]["va"]
    err = np.abs(d["Y"][L] - predict_returns(model, d, L, y_std))
    return {str(c): np.quantile(err, c, axis=0).tolist() for c in config.INTERVAL_LEVELS}


def evaluate(model, d, y_std, q, X=None, L=None):
    L = d["split"]["te"] if L is None else L
    pred = predict_returns(model, d, L, y_std, X)
    close = d["df"]["close"].values
    base = close[L]
    res, rows = {}, []
    for k, h in enumerate(H):
        actual = close[L + h]
        p = base * np.exp(pred[:, k])
        err = np.abs(np.log(actual / p))
        res[f"{h}h"] = {
            "lstm": price_metrics(actual, p),
            "naive_baseline": price_metrics(actual, base),
            "direction_accuracy_%": float(np.mean(np.sign(p - base) == np.sign(actual - base)) * 100),
            "interval_coverage_%": {c: float(np.mean(err <= q[c][k]) * 100) for c in q},
            "interval_half_width_%": {c: float((np.exp(q[c][k]) - 1) * 100) for c in q},
        }
        rows.append(pd.DataFrame({"horizon_h": h, "made_at": d["df"].index[L],
                                  "target_time": d["df"].index[L + h], "actual": actual,
                                  "predicted": p,
                                  "low_95": p * np.exp(-q["0.95"][k]),
                                  "high_95": p * np.exp(q["0.95"][k])}))
    return res, pd.concat(rows, ignore_index=True)


def factor_analysis(model, d, y_std, q, n_sample=4000):
    df, cols = d["df"], d["cols"]
    fwd24 = np.log(df["close"]).shift(-24) - np.log(df["close"])
    train_mask = df.index < df.index[d["split"]["tr"].max()]
    corr = {}
    for c in cols:
        v = df.loc[train_mask, c].corr(fwd24[train_mask])
        g = group_of(c)
        if pd.notna(v) and abs(v) > abs(corr.get(g, 0)):
            corr[g] = float(v)

    rng = np.random.default_rng(0)
    te = d["split"]["te"]
    L = np.sort(rng.choice(te, size=min(n_sample, len(te)), replace=False))
    X = windows(d, L)
    base = evaluate(model, d, y_std, q, X, L)[0]
    groups = {}
    for i, c in enumerate(cols):
        groups.setdefault(group_of(c), []).append(i)
    imp = {}
    for g, idx in groups.items():
        Xp = X.copy()
        Xp[:, :, idx] = X[rng.permutation(len(X))][:, :, idx]
        r = evaluate(model, d, y_std, q, Xp, L)[0]
        imp[g] = {f"{h}h": r[f"{h}h"]["lstm"]["MAE"] - base[f"{h}h"]["lstm"]["MAE"] for h in (1, 24)}
    return {"correlation_with_next_24h_return": corr, "permutation_importance_mae_increase": imp}


def main():
    t0 = time.time()
    print("Downloading gold hourly history (first run can take a while, then it is cached)...")
    close = clean_prices(metal_hourly("gold", config.TRAIN_START_HOURLY))
    close.to_csv(config.history_csv("gold"))
    print(f"Gold hourly rows: {len(close):,} from {close.index[0]} to {close.index[-1]}")

    store = FactorStore("gold")
    store.refresh(config.TRAIN_START_HOURLY, config.TRAIN_START_DAILY)
    print("Driver status:", json.dumps(store.status, indent=1))

    base_df, _ = build_features(close, "gold", None, "base", training=True)
    t = base_df.index
    cut_train, cut_val = t[int(len(t) * 0.8)], t[int(len(t) * 0.9)]
    print(f"Train < {cut_train.date()} <= Validation < {cut_val.date()} <= Test")

    out, trained = {}, {}
    for fs in config.FEATURE_SETS:
        print(f"\n--- feature set: {fs} ---")
        d = prepare(close, store.raw, fs, cut_train, cut_val)
        sp = d["split"]
        print(f"{len(d['cols'])} features | windows: train {len(sp['tr']):,}, "
              f"val {len(sp['va']):,}, test {len(sp['te']):,}")
        model, y_std, val_loss = train_set(d)
        q = interval_quantiles(model, d, y_std)
        res, preds = evaluate(model, d, y_std, q)
        for h in H:
            r = res[f"{h}h"]
            print(f"{h:>2}h  accuracy {r['lstm']['price_accuracy_%']:.3f}% "
                  f"(naive {r['naive_baseline']['price_accuracy_%']:.3f}%)  "
                  f"MAE ${r['lstm']['MAE']:.2f}  direction {r['direction_accuracy_%']:.1f}%  "
                  f"95% band covers {r['interval_coverage_%']['0.95']:.1f}%")
        out[fs] = {"horizons": res, "val_loss": val_loss, "n_features": len(d["cols"]),
                   "features": d["cols"], "test_windows": int(len(sp["te"]))}
        trained[fs] = (model, y_std, d, preds, q)

    selected = min(config.FEATURE_SETS, key=lambda f: out[f]["val_loss"])
    print(f"\nSelected for live use: {selected}")
    model, y_std, d, preds, q = trained[selected]
    model.save(config.model_path("gold"))
    joblib.dump({"x_scaler": d["scaler"], "y_std": y_std, "horizons": H, "features": d["cols"],
                 "feature_set": selected, "interval_q": q}, config.scaler_path("gold"))
    preds.to_csv(config.test_pred_csv("gold"), index=False)
    plot_results(close, preds)

    if "full" in trained:
        fm, fy, fd, _, fq = trained["full"]
        analysis = factor_analysis(fm, fd, fy, fq)
        analysis["factor_status"] = store.status
        config.factor_analysis_json("gold").write_text(json.dumps(analysis, indent=2))

    metrics = {"gold": {"selected": selected, "feature_sets": out,
                        "data_from": str(close.index[0]), "data_to": str(close.index[-1]),
                        "rows": int(len(close)),
                        "split": {"train_until": str(cut_train), "validation_until": str(cut_val)}}}
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to {config.METRICS_JSON}  ({(time.time() - t0) / 60:.1f} min)")


def plot_results(close, out):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(close.index, close.values, lw=0.6)
    ax.set_yscale("log")
    ax.set_title("Gold: hourly spot price since 2003 (USD/oz, log scale)")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "gold_history.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(len(H), 1, figsize=(12, 3 * len(H)), sharex=True)
    for ax, h in zip(axes, H):
        p = out[out["horizon_h"] == h].tail(300)
        ax.fill_between(p["target_time"], p["low_95"], p["high_95"], alpha=0.2, label="95% band")
        ax.plot(p["target_time"], p["actual"], label="Actual", lw=1)
        ax.plot(p["target_time"], p["predicted"], label="Predicted", lw=1, ls="--")
        ax.set_title(f"Gold: {h} hour(s) ahead")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "gold_actual_vs_predicted.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
