"""News layer: world headlines -> Transformer sentiment -> gold/silver news score.

Sources (both free, no key):
  1) GDELT DOC 2.0 API (main)
  2) Google News RSS (backup)

Sentiment model: FinBERT (ProsusAI/finbert), a Transformer trained on financial text.

IMPORTANT: the news score is NOT learned from data. It follows well-known market
logic (safe-haven demand, interest-rate effect). It is shown as a separate
"news-adjusted scenario" next to the pure LSTM forecast.
"""
import logging
import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

import config

log = logging.getLogger("news")


# ---------- 1) collect headlines ----------
def _from_gdelt(query: str) -> list:
    params = {"query": f"{query} sourcelang:english", "mode": "ArtList", "format": "json",
              "maxrecords": config.NEWS_PER_TOPIC, "timespan": "24h", "sort": "DateDesc"}
    r = requests.get(config.GDELT_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()                         # raises ValueError if GDELT sent text
    return [{"title": a.get("title", "").strip(), "url": a.get("url"),
             "source": a.get("domain"), "published": a.get("seendate")}
            for a in data.get("articles", []) if a.get("title")]


def _from_google_rss(query: str) -> list:
    q = query.replace('"', "").replace("(", "").replace(")", "")
    params = {"q": f"{q} when:1d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    r = requests.get(config.GOOGLE_NEWS_RSS, params=params, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        items.append({"title": (it.findtext("title") or "").strip(),
                      "url": it.findtext("link"),
                      "source": it.findtext("source"),
                      "published": it.findtext("pubDate")})
        if len(items) >= config.NEWS_PER_TOPIC:
            break
    return items


def fetch_topic(query: str) -> tuple:
    """Returns (headlines, source_name). Never raises."""
    try:
        items = _from_gdelt(query)
        if items:
            return items, "GDELT"
    except Exception as e:
        log.warning("GDELT failed: %s", e)
    try:
        return _from_google_rss(query), "Google News RSS"
    except Exception as e:
        log.warning("Google News RSS failed: %s", e)
        return [], "none"


# ---------- 2) sentiment ----------
class Sentiment:
    POS = ["surge", "gain", "rise", "growth", "peace", "deal", "agreement", "ceasefire", "recover"]
    NEG = ["war", "attack", "strike", "crisis", "fall", "drop", "sanction", "tension",
           "conflict", "threat", "escalat", "recession", "tariff"]

    def __init__(self):
        self.pipe = None
        try:
            from transformers import pipeline
            self.pipe = pipeline("text-classification", model=config.SENTIMENT_MODEL, top_k=None)
            log.info("Sentiment model loaded: %s", config.SENTIMENT_MODEL)
        except Exception as e:
            log.warning("FinBERT not loaded, using word list instead: %s", e)

    def score(self, titles: list) -> list:
        """For each title: net sentiment in [-1, 1] (positive minus negative)."""
        if not titles:
            return []
        if self.pipe is not None:
            try:
                out = self.pipe(titles, truncation=True, batch_size=16)
                res = []
                for labels in out:
                    d = {x["label"].lower(): x["score"] for x in labels}
                    res.append(d.get("positive", 0) - d.get("negative", 0))
                return res
            except Exception as e:
                log.warning("FinBERT scoring failed: %s", e)
        res = []
        for t in titles:
            t = t.lower()
            p = sum(w in t for w in self.POS)
            n = sum(w in t for w in self.NEG)
            res.append(0.0 if p == n == 0 else (p - n) / (p + n))
        return res


# ---------- 3) turn sentiment into a gold signal ----------
def gold_impact(title: str, sentiment: float, effect: str) -> float:
    """+1 = supports higher gold price, -1 = supports lower gold price."""
    t = title.lower()
    if effect == "policy":
        if any(w in t for w in config.DOVISH_WORDS):
            return 1.0
        if any(w in t for w in config.HAWKISH_WORDS):
            return -1.0
        return 0.0
    if effect == "risk":
        return -sentiment          # bad news for the world -> safe-haven buying
    return sentiment               # "direct": good news about gold -> up


def build_news_state(sentiment: Sentiment) -> dict:
    topics, total, used_weight = {}, 0.0, 0.0
    for i, (key, cfg) in enumerate(config.NEWS_TOPICS.items()):
        if i:
            time.sleep(6)          # GDELT asks for max 1 request every 5 seconds
        items, src = fetch_topic(cfg["query"])
        scores = sentiment.score([x["title"] for x in items])
        for x, s in zip(items, scores):
            x["sentiment"] = round(float(s), 3)
            x["gold_impact"] = round(gold_impact(x["title"], s, cfg["effect"]), 3)
        topic_score = (sum(x["gold_impact"] for x in items) / len(items)) if items else None
        topics[key] = {"label": cfg["label"], "score": topic_score, "source": src,
                       "count": len(items), "headlines": items}
        if topic_score is not None:
            total += cfg["weight"] * topic_score
            used_weight += cfg["weight"]

    overall = total / used_weight if used_weight else None
    return {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "overall_score": overall, "label": describe(overall), "topics": topics}


def describe(score) -> str:
    if score is None:
        return "No news available"
    if score > 0.15:
        return "News supports a HIGHER gold price"
    if score < -0.15:
        return "News supports a LOWER gold price"
    return "News is mixed or neutral for gold"


def adjust_price(now: float, lstm_price: float, hours: int, score, hourly_vol: float,
                 metal: str) -> float:
    """Shift the LSTM return by a small amount based on the news score.
    shift = strength x score x hourly volatility x sqrt(hours)
    so the effect grows with the horizon but stays within normal market moves."""
    if score is None or not hourly_vol or math.isnan(hourly_vol):
        return lstm_price
    factor = 1.0 if metal == "gold" else config.NEWS_SILVER_FACTOR
    shift = config.NEWS_ADJUST_STRENGTH * factor * score * hourly_vol * math.sqrt(hours)
    return float(now * math.exp(math.log(lstm_price / now) + shift))
