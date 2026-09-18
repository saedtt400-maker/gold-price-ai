"""FastAPI backend.

External APIs -> FastAPI -> Data Processing -> LSTM -> Prediction
                                 + News (GDELT + FinBERT)  -> Streamlit + Chatbot

Run from the project folder:
    uvicorn backend.main:app --port 8000
"""
import asyncio
import csv
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from backend import news as news_mod
from backend.chatbot import Chatbot
from backend.factors import FactorStore, specs_for
from backend.predictor import MetalPredictor, PredictionError
from backend.history_sources import metal_hourly
from backend.price_client import ExternalAPIError, fetch_live_price, fetch_usd_ils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backend")

STATE = {
    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "usd_ils": None,
    "news": {"updated_at": None, "overall_score": None,
             "label": "News is loading...", "topics": {}},
    "metals": {m: {"price_usd_oz": None, "predictions": {}, "last_update": None,
                   "stats": None, "error": None, "ticks": []} for m in config.METALS},
}
HISTORY, BASIS, PREDICTORS, MODEL_ERRORS = {}, {}, {}, {}
FACTORS = {m: FactorStore(m) for m in config.METALS}
CHATBOT = None
SENTIMENT = None
_last_history_refresh = None


def _now():
    return datetime.now(timezone.utc)


# ---------- prices ----------
def _live_start() -> str:
    return (_now() - pd.Timedelta(days=config.LIVE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")


def refresh_history():
    """Recent hourly gold (Dukascopy + Yahoo) and drivers on their own schedules."""
    global _last_history_refresh
    start = _live_start()
    for metal in config.METALS:
        try:
            close = metal_hourly(metal, start)
            live = fetch_live_price(metal)["price_usd_oz"]
            HISTORY[metal] = close
            BASIS[metal] = float(close.iloc[-1]) / live   # history source vs live spot
        except (ExternalAPIError, IndexError) as e:
            log.warning("History refresh failed for %s: %s", metal, e)
        fs = FACTORS[metal]
        groups = ["hourly"]
        if fs.due("daily", config.DAILY_FACTOR_REFRESH_SECONDS):
            groups.append("daily")
        if fs.due("slow", config.SLOW_FACTOR_REFRESH_SECONDS):
            groups.append("slow")
        daily_start = (_now() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        fs.refresh(start, daily_start, groups)
    try:
        STATE["usd_ils"] = fetch_usd_ils()
    except ExternalAPIError as e:
        log.warning("%s", e)
    _last_history_refresh = _now()


def series_with_live(metal: str, live_price: float) -> pd.Series:
    close = HISTORY[metal]
    now = pd.Timestamp(_now())
    finished = close[close.index <= now]          # index = bar close time
    return pd.concat([finished, pd.Series([live_price * BASIS.get(metal, 1.0)], index=[now])])


def hourly_vol(metal: str) -> float:
    r = np.log(HISTORY[metal]).diff().tail(24)
    return float(r.std())


def log_tick(row: dict):
    new = not config.LIVE_LOG_CSV.exists()
    with open(config.LIVE_LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def poll_once():
    if _last_history_refresh is None or \
            (_now() - _last_history_refresh).total_seconds() > config.HISTORY_REFRESH_SECONDS:
        refresh_history()

    news_score = STATE["news"]["overall_score"]
    for metal in config.METALS:
        s = STATE["metals"][metal]
        try:
            price = fetch_live_price(metal)["price_usd_oz"]
        except ExternalAPIError as e:
            s["error"] = f"Live price unavailable: {e}. Showing last known value."
            log.warning("%s: %s", metal, e)
            continue

        ts = _now().isoformat(timespec="seconds")
        s.update(price_usd_oz=price, last_update=ts, error=None)
        s["ticks"] = (s["ticks"] + [{"time": ts, "price": price}])[-500:]

        preds = {}
        if metal in PREDICTORS and metal in HISTORY:
            try:
                b = BASIS.get(metal, 1.0)
                raw = PREDICTORS[metal].predict(series_with_live(metal, price), FACTORS[metal].raw)
                if PREDICTORS[metal].missing:
                    s["error"] = ("Some market drivers are unavailable, their average is used: "
                                  + ", ".join(sorted({c.split('__')[0] for c in PREDICTORS[metal].missing})))
                vol = hourly_vol(metal)
                for h, v in raw.items():
                    lstm_p = v["price"] / b
                    adj = news_mod.adjust_price(price, lstm_p, h, news_score, vol, metal)
                    preds[f"{h}h"] = {
                        "hours": h,
                        "lstm": lstm_p,
                        "news_adjusted": adj,
                        # bands are relative to the forecast, so they move with the news shift too
                        "bands": {c: [lo / b, hi / b] for c, (lo, hi) in v["bands"].items()},
                        "bands_news": {c: [lo / v["price"] * adj, hi / v["price"] * adj]
                                       for c, (lo, hi) in v["bands"].items()},
                    }
            except PredictionError as e:
                s["error"] = str(e)
        elif metal in MODEL_ERRORS:
            s["error"] = MODEL_ERRORS[metal]
        s["predictions"] = preds

        if metal in HISTORY:
            h24 = HISTORY[metal]
            last24 = h24[h24.index > h24.index[-1] - pd.Timedelta(hours=24)] / BASIS.get(metal, 1.0)
            s["stats"] = {"min_24h": float(last24.min()), "max_24h": float(last24.max()),
                          "mean_24h": float(last24.mean()),
                          "change_24h_pct": float((price / last24.iloc[0] - 1) * 100)}

        row = {"time": ts, "metal": metal, "price_usd_oz": price,
               "usd_ils": STATE["usd_ils"], "news_score": news_score}
        for h in config.HORIZONS:
            row[f"lstm_{h}h"] = preds.get(f"{h}h", {}).get("lstm")
        log_tick(row)


async def price_loop():
    while True:
        try:
            await asyncio.to_thread(poll_once)
        except Exception as e:
            log.exception("Price cycle failed: %s", e)
        await asyncio.sleep(config.POLL_SECONDS)


# ---------- news ----------
async def news_loop():
    while True:
        try:
            STATE["news"] = await asyncio.to_thread(news_mod.build_news_state, SENTIMENT)
            log.info("News updated: %s", STATE["news"]["label"])
        except Exception as e:
            log.exception("News cycle failed: %s", e)
            STATE["news"]["label"] = "News is not available right now."
        await asyncio.sleep(config.NEWS_REFRESH_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CHATBOT, SENTIMENT
    for metal in config.METALS:
        try:
            PREDICTORS[metal] = MetalPredictor(metal)
        except PredictionError as e:
            MODEL_ERRORS[metal] = str(e)
            log.warning("%s", e)
    SENTIMENT = await asyncio.to_thread(news_mod.Sentiment)
    CHATBOT = await asyncio.to_thread(Chatbot)
    tasks = [asyncio.create_task(price_loop()), asyncio.create_task(news_loop())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Gold & Silver Real-Time AI API", lifespan=lifespan)


def _check_metal(metal: str):
    if metal not in config.METALS:
        raise HTTPException(404, f"Unknown metal '{metal}'. Use: {list(config.METALS)}")


def snapshot() -> dict:
    return {
        "usd_ils": STATE["usd_ils"],
        "news": {k: v for k, v in STATE["news"].items() if k != "topics"}
                | {"topics": {k: {"label": t["label"], "score": t["score"],
                                  "top": [h["title"] for h in t["headlines"][:3]]}
                              for k, t in STATE["news"]["topics"].items()}},
        "metals": {m: {k: v for k, v in s.items() if k != "ticks"}
                   for m, s in STATE["metals"].items()},
        "drivers": {m: driver_rows(m) for m in config.METALS},
    }


def driver_rows(metal: str) -> list:
    latest = FACTORS[metal].latest()
    analysis = {}
    path = config.factor_analysis_json(metal)
    if path.exists():
        analysis = json.loads(path.read_text())
    corr = analysis.get("correlation_with_next_24h_return", {})
    imp = analysis.get("permutation_importance_mae_increase", {})
    used = set()
    if metal in PREDICTORS and PREDICTORS[metal].feature_set == "full":
        used = {c.split("__")[0] for c in PREDICTORS[metal].features if "__" in c}
    rows = []
    for key, spec in specs_for(metal).items():
        lt = latest.get(key, {})
        rows.append({
            "key": key, "label": spec["label"], "category": spec["category"],
            "frequency": spec["freq"], "expected": spec["relation"],
            "status": FACTORS[metal].status.get(key, "not loaded"),
            "value": lt.get("value"), "change_24h": lt.get("change"), "as_of": lt.get("as_of"),
            "corr_next_24h": corr.get(key), "importance_24h": imp.get(key, {}).get("24h"),
            "in_live_model": key in used,
        })
    return rows


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "started_at": STATE["started_at"],
            "models_loaded": {m: p.feature_set for m, p in PREDICTORS.items()},
            "model_errors": MODEL_ERRORS,
            "factor_status": {m: f.status for m, f in FACTORS.items()},
            "sentiment_model": SENTIMENT is not None and SENTIMENT.pipe is not None,
            "chatbot": CHATBOT is not None and CHATBOT.pipe is not None,
            "horizons": config.HORIZONS}


@app.get("/latest")
def latest():
    snap = snapshot()
    return {"server_time": _now().isoformat(timespec="seconds"),
            "horizons": config.HORIZONS, "usd_ils": snap["usd_ils"],
            "news_summary": {"score": STATE["news"]["overall_score"],
                             "label": STATE["news"]["label"],
                             "updated_at": STATE["news"]["updated_at"]},
            "metals": snap["metals"]}


@app.get("/news")
def news():
    return STATE["news"]


@app.get("/factors/{metal}")
def factors(metal: str):
    _check_metal(metal)
    own = {}
    path = config.factor_analysis_json(metal)
    if path.exists():
        a = json.loads(path.read_text())
        own = {"corr_next_24h": a.get("correlation_with_next_24h_return", {}).get("own_price"),
               "importance": a.get("permutation_importance_mae_increase", {}).get("own_price")}
    return {"feature_set": PREDICTORS[metal].feature_set if metal in PREDICTORS else None,
            "drivers": driver_rows(metal),
            "own_price_technical": own,
            "context_only": [{"factor": f, "reason": r} for f, m, r in config.CONTEXT_ONLY if m == metal]}


@app.get("/ticks/{metal}")
def ticks(metal: str):
    _check_metal(metal)
    return STATE["metals"][metal]["ticks"]


@app.get("/history/{metal}")
def history(metal: str, hours: int = 168):
    _check_metal(metal)
    if metal not in HISTORY:
        raise HTTPException(503, "History is not loaded yet. Try again in a few seconds.")
    close = HISTORY[metal].tail(hours) / BASIS.get(metal, 1.0)
    return [{"time": t.isoformat(), "price": float(p)} for t, p in close.items()]


@app.get("/comparison/{metal}")
def comparison(metal: str, hours: int = 72):
    _check_metal(metal)
    if metal not in PREDICTORS:
        raise HTTPException(503, MODEL_ERRORS.get(metal, "Model not loaded."))
    if metal not in HISTORY:
        raise HTTPException(503, "History is not loaded yet.")
    try:
        df = PREDICTORS[metal].rolling_predictions(HISTORY[metal], FACTORS[metal].raw, hours)
    except PredictionError as e:
        raise HTTPException(500, str(e))
    b = BASIS.get(metal, 1.0)
    return [{"time": r.time.isoformat(), "actual": r.actual / b, "predicted": r.predicted / b}
            for r in df.itertuples()]


@app.get("/metrics")
def metrics():
    if not config.METRICS_JSON.exists():
        raise HTTPException(404, "No metrics yet. Train the models first.")
    return json.loads(config.METRICS_JSON.read_text())


@app.get("/test-predictions/{metal}")
def test_predictions(metal: str, horizon: int = 1, last: int = 300):
    _check_metal(metal)
    path = config.test_pred_csv(metal)
    if not path.exists():
        raise HTTPException(404, "No test predictions yet. Train the models first.")
    df = pd.read_csv(path)
    df = df[df["horizon_h"] == horizon].tail(last)
    return df.to_dict(orient="records")


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest):
    if CHATBOT is None:
        raise HTTPException(503, "Chatbot is still loading.")
    return await asyncio.to_thread(CHATBOT.answer, req.question, snapshot())
