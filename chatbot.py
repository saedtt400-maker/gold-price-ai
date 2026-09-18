"""Transformer chatbot (Flan-T5) that answers questions about the live data and news."""
import logging

import config

log = logging.getLogger("chatbot")
DRIVER_WORDS = ["dollar", "dxy", "yield", "rate", "inflation", "cpi", "oil", "vix", "copper",
                "ratio", "driver", "factor", "stock", "s&p", "etf", "cot", "euro", "yen", "risk"]
NEWS_WORDS = ["news", "war", "peace", "fed", "federal", "central bank", "interest", "china",
              "america", "usa", "middle east", "iran", "israel", "gaza", "why", "reason"]


def _metals_in(question: str) -> list:
    q = question.lower()
    found = [m for m in config.METALS if m in q]
    return found or list(config.METALS)


def build_context(snapshot: dict, question: str = "") -> str:
    """Short factual sentences. Only the metals and parts the question needs,
    because Flan-T5 reads at most about 512 tokens."""
    lines = []
    rate = snapshot.get("usd_ils")
    g = rate / config.GRAMS_PER_OUNCE if rate else None
    q = question.lower()

    for metal in _metals_in(question):
        s = snapshot.get("metals", {}).get(metal, {})
        name = config.METALS[metal]["name"].lower()
        p = s.get("price_usd_oz")
        if p is None:
            lines.append(f"The {name} price is not available right now.")
            continue
        lines.append(f"The current {name} price is {p:,.2f} dollars per ounce"
                     + (f", or {p * g:,.2f} shekels per gram." if g else "."))
        if g and metal == "gold":
            lines.append(f"In {config.LOCAL_MARKET}, 21K gold is {p * g * 21 / 24:,.2f} "
                         f"shekels per gram and 18K is {p * g * 18 / 24:,.2f}.")

        for key, v in s.get("predictions", {}).items():
            lstm, adj = v["lstm"], v["news_adjusted"]
            trend = "rise" if lstm > p else "fall"
            txt = (f"After {v['hours']} hours the {name} price is expected to {trend} to "
                   f"{lstm:,.2f} dollars per ounce ({(lstm - p) / p * 100:+.2f} percent)")
            if g:
                txt += f", which is {lstm * g:,.2f} shekels per gram"
            b95 = v.get("bands", {}).get("0.95")
            if b95:
                txt += (f". The 95 percent range is {b95[0]:,.2f} to {b95[1]:,.2f} dollars")
            txt += f". With news included it is {adj:,.2f} dollars."
            lines.append(txt)

        st = s.get("stats")
        if st:
            lines.append(f"In the last 24 hours {name} ranged from {st['min_24h']:,.2f} to "
                         f"{st['max_24h']:,.2f} dollars, average {st['mean_24h']:,.2f}, "
                         f"change {st['change_24h_pct']:+.2f} percent.")
        if s.get("last_update"):
            lines.append(f"The last {name} update was at {s['last_update']} UTC.")

    if any(w in q for w in DRIVER_WORDS):
        for metal in _metals_in(question):
            for d in snapshot.get("drivers", {}).get(metal, []):
                if d["value"] is None:
                    continue
                ch = ""
                if d["change_24h"] is not None:
                    ch = f", 24 hour change {d['change_24h']:+.2f}"
                lines.append(f"{d['label']} is {d['value']:,.2f}{ch}; "
                             f"its usual relation with {metal} is {d['expected'].lower()}.")

    news = snapshot.get("news", {})
    if news.get("overall_score") is not None:
        lines.append(f"News summary: {news['label']} (score {news['overall_score']:+.2f}).")
        if any(w in q for w in NEWS_WORDS):
            for t in news.get("topics", {}).values():
                if t["score"] is None:
                    continue
                effect = "positive" if t["score"] > 0.1 else "negative" if t["score"] < -0.1 else "neutral"
                head = f" Top headline: {t['top'][0]}." if t.get("top") else ""
                lines.append(f"{t['label']} news is {effect} for gold.{head}")
    if rate:
        lines.append(f"One US dollar equals {rate:.3f} shekels.")
    return " ".join(lines)


class Chatbot:
    def __init__(self):
        self.pipe = None
        try:
            from transformers import pipeline
            self.pipe = pipeline("text2text-generation", model=config.CHAT_MODEL)
            log.info("Chatbot model loaded: %s", config.CHAT_MODEL)
        except Exception as e:
            log.warning("Chatbot model not loaded: %s", e)

    def answer(self, question: str, snapshot: dict) -> dict:
        question = (question or "").strip()
        if not question:
            return {"answer": "Please type a question.", "source": "system"}
        context = build_context(snapshot, question)
        if not context:
            return {"answer": "No data has been collected yet. Please wait a few seconds.",
                    "source": "system"}
        if self.pipe is None:
            return {"answer": self._fallback(question, snapshot), "source": "fallback"}
        prompt = ("You are an assistant for a gold and silver price dashboard. "
                  "Answer in one or two full sentences using only these facts. "
                  "If the facts do not contain the answer, say you do not know.\n\n"
                  f"Facts: {context}\n\nQuestion: {question}\nAnswer:")
        try:
            out = self.pipe(prompt, max_new_tokens=100, do_sample=False,
                            truncation=True)[0]["generated_text"]
            return {"answer": out.strip(), "source": config.CHAT_MODEL}
        except Exception as e:
            log.warning("Chatbot generation failed: %s", e)
            return {"answer": self._fallback(question, snapshot), "source": "fallback"}

    @staticmethod
    def _fallback(question: str, snapshot: dict) -> str:
        """Backup answer used only if the Transformer cannot run."""
        q = question.lower()
        metal = "silver" if "silver" in q else "gold"
        s = snapshot.get("metals", {}).get(metal, {})
        p = s.get("price_usd_oz")
        if p is None:
            return f"{metal.title()} data is not available right now."
        if any(w in q for w in NEWS_WORDS):
            return snapshot.get("news", {}).get("label", "News is not available.")
        for h in sorted(config.HORIZONS, reverse=True):
            if f"{h}" in q and f"{h}h" in s.get("predictions", {}):
                v = s["predictions"][f"{h}h"]
                return (f"{metal.title()} is {p:,.2f} USD/oz now. After {h} hours the LSTM "
                        f"expects {v['lstm']:,.2f}, or {v['news_adjusted']:,.2f} with news.")
        return f"The current {metal} price is {p:,.2f} USD per ounce."
