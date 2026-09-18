"""Streamlit dashboard.

Run from the project folder:
    streamlit run frontend/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

API = config.BACKEND_URL
st.set_page_config(page_title="Gold & Silver AI", page_icon="🪙", layout="wide")


def get(path, **params):
    """Call FastAPI. Returns (data, error_message)."""
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=10)
        if r.status_code != 200:
            try:
                return None, r.json().get("detail", r.text)
            except ValueError:
                return None, r.text
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the FastAPI backend. Is it running on port 8000?"
    except requests.exceptions.Timeout:
        return None, "The backend is slow to answer. Retrying in 10 seconds."


# ---------- sidebar ----------
st.sidebar.title("Settings")
metal = st.sidebar.radio("Metal", list(config.METALS),
                         format_func=lambda m: config.METALS[m]["name"])
currency = st.sidebar.radio("Currency", ["ILS (₪)", "USD ($)"])
unit = st.sidebar.radio("Show price per", ["Gram", "Ounce"])
use_news = st.sidebar.toggle("Include news in the forecast", value=True)
band = st.sidebar.select_slider("Confidence band", options=["0.8", "0.95", "0.99"], value="0.95",
                                format_func=lambda c: f"{float(c) * 100:.0f}%")
hours = st.sidebar.slider("History window (hours)", 24, 24 * 30, 24 * 7, step=24)
st.sidebar.markdown(f"**Local market: {config.LOCAL_MARKET}**")
making = st.sidebar.number_input("Workmanship (مصنعية), ₪ per gram", 0.0, 200.0, 0.0, step=5.0,
                                 help="Shop fee added on top of the metal price. Set 0 for raw price.")
st.sidebar.caption(f"Prices refresh every {config.POLL_SECONDS} s, "
                   f"news every {config.NEWS_REFRESH_SECONDS // 60} min.")

name = config.METALS[metal]["name"]
IS_ILS = currency.startswith("ILS")
CUR = "ILS" if IS_ILS else "USD"
SIGN = "₪" if IS_ILS else "$"
RATE = {"value": None}
PRED_KEY = "news_adjusted" if use_news else "lstm"
BAND_KEY = "bands_news" if use_news else "bands"
BAND_TXT = f"{float(band) * 100:.0f}%"


def to_unit(usd_oz):
    v = usd_oz / config.GRAMS_PER_OUNCE if unit == "Gram" else usd_oz
    if IS_ILS:
        v = v * RATE["value"] if RATE["value"] else float("nan")
    return v


def money(usd_oz):
    return f"{SIGN}{to_unit(usd_oz):,.2f}"


def small_layout(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title=f"{CUR} per {unit.lower()}")
    return fig


st.title(f"🪙 {name}: live price and forecasts up to 24 hours ({CUR})")


# ---------- live section ----------
@st.fragment(run_every=config.POLL_SECONDS)
def live_section():
    data, err = get("/latest")
    if err:
        st.error(err)
        return
    s = data["metals"][metal]
    rate = data.get("usd_ils")
    RATE["value"] = rate
    if IS_ILS and not rate:
        st.warning("USD/ILS rate is not loaded yet, so shekel values are not available.")
    if s.get("error"):
        st.warning(s["error"])
    if s.get("price_usd_oz") is None:
        st.info("Waiting for the first live price...")
        return

    price, preds = s["price_usd_oz"], s.get("predictions", {})
    ns = data.get("news_summary", {})

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Current price ({CUR}/{unit.lower()})", money(price))
    if rate:
        c2.metric("USD/ILS rate", f"1 $ = ₪{rate:.3f}")
    score = ns.get("score")
    c3.metric("News signal for gold", "n/a" if score is None else f"{score:+.2f}",
              help=ns.get("label"))
    st.caption(f"Last update: {s['last_update']} UTC  |  News: {ns.get('label')}")

    # --- forecast table for all horizons
    st.subheader("🔮 Expected price")
    if preds:
        rows = []
        for key, v in preds.items():
            lo, hi = v.get(BAND_KEY, {}).get(band, [None, None])
            rows.append({
                "After": key,
                f"LSTM ({CUR}/{unit.lower()})": to_unit(v["lstm"]),
                f"LSTM + news ({CUR}/{unit.lower()})": to_unit(v["news_adjusted"]),
                f"{BAND_TXT} low": to_unit(lo) if lo else None,
                f"{BAND_TXT} high": to_unit(hi) if hi else None,
                "± %": (hi / v[PRED_KEY] - 1) * 100 if hi else None,
                "Change %": (v[PRED_KEY] - price) / price * 100,
                "Direction": "⬆ Up" if v[PRED_KEY] > price else "⬇ Down",
            })
        tbl = pd.DataFrame(rows).set_index("After")
        st.dataframe(tbl.style.format({c: "{:,.2f}" for c in tbl.columns if c != "Direction"}
                                      | {"Change %": "{:+.3f}", "± %": "±{:.2f}"}, na_rep="–"),
                     use_container_width=True)
        st.caption(f"Change % and Direction use the {'news-adjusted' if use_news else 'pure LSTM'} "
                   f"forecast. The {BAND_TXT} band is the range that held the real price in about "
                   f"{BAND_TXT} of past cases (see Model evaluation).")
    else:
        st.info("No forecast yet.")

    # --- Ramallah table
    if rate and preds:
        st.subheader(f"🏪 Price per gram in {config.LOCAL_MARKET} (₪)")
        purities = config.GOLD_KARATS if metal == "gold" else config.SILVER_PURITY
        g = rate / config.GRAMS_PER_OUNCE
        rows = []
        for label, purity in purities.items():
            row = {"Karat / purity": label, "Now": price * g * purity,
                   "Now + workmanship": price * g * purity + making}
            for key, v in preds.items():
                row[f"After {key}"] = v[PRED_KEY] * g * purity
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Karat / purity").style.format("{:,.2f}"),
                     use_container_width=True)
        st.caption("Raw metal value. Shop prices add workmanship and may differ slightly.")

    left, right = st.columns(2)
    with left:
        st.subheader("Live prices (every 10 s)")
        ticks, e = get(f"/ticks/{metal}")
        if e:
            st.warning(e)
        elif ticks:
            t = pd.DataFrame(ticks)
            fig = go.Figure(go.Scatter(x=t["time"], y=t["price"].map(to_unit), mode="lines+markers"))
            st.plotly_chart(small_layout(fig), use_container_width=True)

    with right:
        st.subheader("History and forecast path")
        hist, e = get(f"/history/{metal}", hours=hours)
        if e:
            st.warning(e)
        elif hist:
            h = pd.DataFrame(hist)
            fig = go.Figure(go.Scatter(x=h["time"], y=h["price"].map(to_unit),
                                       name="History", mode="lines"))
            if preds:
                now = pd.Timestamp(data["server_time"])
                xs = [now] + [now + pd.Timedelta(hours=v["hours"]) for v in preds.values()]
                bl = [to_unit(price)] + [to_unit(v.get(BAND_KEY, {}).get(band, [v[PRED_KEY]] * 2)[0])
                                         for v in preds.values()]
                bh = [to_unit(price)] + [to_unit(v.get(BAND_KEY, {}).get(band, [v[PRED_KEY]] * 2)[1])
                                         for v in preds.values()]
                fig.add_trace(go.Scatter(x=xs + xs[::-1], y=bh + bl[::-1], fill="toself",
                                         line=dict(width=0), opacity=0.2,
                                         name=f"{BAND_TXT} band"))
                for key, label, style in [("lstm", "LSTM", "dash"),
                                          ("news_adjusted", "LSTM + news", "dot")]:
                    ys = [to_unit(price)] + [to_unit(v[key]) for v in preds.values()]
                    fig.add_trace(go.Scatter(x=xs, y=ys, name=label,
                                             mode="lines+markers", line=dict(dash=style)))
            st.plotly_chart(small_layout(fig), use_container_width=True)

    st.subheader("Actual vs predicted (last 72 hours, 1 hour ahead)")
    comp, e = get(f"/comparison/{metal}", hours=72)
    if e:
        st.warning(e)
    elif comp:
        c = pd.DataFrame(comp)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c["time"], y=c["actual"].map(to_unit), name="Actual"))
        fig.add_trace(go.Scatter(x=c["time"], y=c["predicted"].map(to_unit),
                                 name="LSTM predicted", line=dict(dash="dash")))
        st.plotly_chart(small_layout(fig, 340), use_container_width=True)


live_section()


# ---------- news section ----------
@st.fragment(run_every=60)
def news_section():
    st.divider()
    st.subheader("🌍 World news and its effect on gold")
    nd, e = get("/news")
    if e:
        st.warning(e)
        return
    if not nd.get("topics"):
        st.info(nd.get("label", "News is loading..."))
        return
    st.write(f"**{nd['label']}**  (overall score {nd['overall_score']:+.2f}, "
             f"updated {nd['updated_at']} UTC)")

    topics = nd["topics"]
    labels = [t["label"] for t in topics.values()]
    scores = [t["score"] if t["score"] is not None else 0 for t in topics.values()]
    fig = go.Figure(go.Bar(x=scores, y=labels, orientation="h",
                           marker_color=["seagreen" if v > 0 else "indianred" for v in scores]))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(range=[-1, 1], title="← lower gold   |   higher gold →"))
    st.plotly_chart(fig, use_container_width=True)

    for key, t in topics.items():
        sc = "n/a" if t["score"] is None else f"{t['score']:+.2f}"
        with st.expander(f"{t['label']}  |  score {sc}  |  {t['count']} headlines ({t['source']})"):
            if not t["headlines"]:
                st.write("No headlines found.")
            for hl in t["headlines"][:10]:
                icon = "🟢" if hl["gold_impact"] > 0.1 else "🔴" if hl["gold_impact"] < -0.1 else "⚪"
                st.markdown(f"{icon} [{hl['title']}]({hl['url']})  \n"
                            f"<small>{hl.get('source') or ''} · impact {hl['gold_impact']:+.2f}</small>",
                            unsafe_allow_html=True)
    st.caption("🟢 supports a higher gold price · 🔴 supports a lower price. "
               "Sentiment by FinBERT (Transformer). The news effect is a rule-based scenario, "
               "not learned from past data.")


news_section()

# ---------- market drivers ----------
@st.fragment(run_every=60)
def drivers_section():
    st.divider()
    st.subheader(f"📈 Market drivers for {name}")
    fd, e = get(f"/factors/{metal}")
    if e:
        st.warning(e)
        return
    fs = fd.get("feature_set")
    if fs:
        st.write("Live model uses: **" + ("price + technical + market drivers" if fs == "full"
                                          else "price + technical only") + "**")
    rows = []
    for d in fd["drivers"]:
        rows.append({
            "Category": d["category"], "Factor": d["label"], "Updates": d["frequency"],
            "Latest": d["value"], "Change (24h or last day)": d["change_24h"],
            "Expected relation": d["expected"],
            "Corr. with next 24h return": d["corr_next_24h"],
            "Importance (24h MAE +$)": d["importance_24h"],
            "Status": "✅" if d["status"] == "ok" else "⚠️ " + str(d["status"])[:60],
        })
    if rows:
        df = pd.DataFrame(rows)
        num = ["Latest", "Change (24h or last day)", "Corr. with next 24h return", "Importance (24h MAE +$)"]
        st.dataframe(df.style.format({c: "{:,.3f}" for c in num}, na_rep="–"),
                     use_container_width=True, hide_index=True)
        imp = df.dropna(subset=["Importance (24h MAE +$)"])
        own = (fd.get("own_price_technical") or {}).get("importance") or {}
        if not imp.empty:
            bars = imp[["Factor", "Importance (24h MAE +$)"]].copy()
            if own.get("24h") is not None:
                bars.loc[len(bars)] = [f"{name} own price + technical", own["24h"]]
            bars = bars.sort_values("Importance (24h MAE +$)")
            fig = go.Figure(go.Bar(x=bars["Importance (24h MAE +$)"], y=bars["Factor"],
                                   orientation="h"))
            fig.update_layout(height=30 * len(bars) + 80, margin=dict(l=10, r=10, t=30, b=10),
                              title="Permutation importance: how much the 24h error grows "
                                    "when the factor is shuffled")
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Technical indicators inside the model: previous prices, returns (1h, 4h, 24h), "
               "volatility, RSI(14), MACD histogram, 24h and 120h moving averages.")
    if fd.get("context_only"):
        with st.expander("Factors not in the hourly model (no free or frequent data)"):
            st.dataframe(pd.DataFrame(fd["context_only"]), use_container_width=True, hide_index=True)


drivers_section()

# ---------- model evaluation ----------
with st.expander("📊 Model evaluation (test set, USD per ounce)", expanded=False):
    metrics, e = get("/metrics")
    if e:
        st.warning(e)
    else:
        m = metrics.get(metal, {})
        st.write(f"Data: **{m.get('rows', 0):,} hourly prices** from {str(m.get('data_from'))[:10]} "
                 f"to {str(m.get('data_to'))[:10]}. Selected for live use: **{m.get('selected', '?')}**")
        rows = []
        for fs_name, fs_val in m.get("feature_sets", {}).items():
            for key, v in fs_val["horizons"].items():
                cov = v.get("interval_coverage_%", {})
                wid = v.get("interval_half_width_%", {})
                rows.append({
                    "Horizon": key, "Features": f"{fs_name} ({fs_val['n_features']})",
                    "Price accuracy %": v["lstm"].get("price_accuracy_%"),
                    "Naive accuracy %": v["naive_baseline"].get("price_accuracy_%"),
                    "LSTM MAE $": v["lstm"]["MAE"], "Naive MAE $": v["naive_baseline"]["MAE"],
                    "RMSE $": v["lstm"]["RMSE"], "R²": v["lstm"]["R2"],
                    "Direction %": v["direction_accuracy_%"],
                    f"{BAND_TXT} band ±%": wid.get(band),
                    f"{BAND_TXT} band hit %": cov.get(band),
                })
        if rows:
            t = pd.DataFrame(rows).sort_values(["Horizon", "Features"],
                                               key=lambda c: c.str.extract(r"(\d+)")[0].astype(float)
                                               if c.name == "Horizon" else c)
            st.dataframe(t.style.format({c: "{:,.3f}" for c in t.columns
                                         if c not in ("Horizon", "Features")}, na_rep="–"),
                         use_container_width=True, hide_index=True)
            st.caption("Price accuracy = 100 − MAPE. Compare it with the naive model "
                       "(price stays the same). Direction % is the share of correct up/down calls. "
                       "Band hit % shows how often the real price fell inside the band on test data.")
    hsel = st.selectbox("Horizon for the test chart", config.HORIZONS,
                        format_func=lambda h: f"{h} hour(s)")
    tp, e = get(f"/test-predictions/{metal}", horizon=hsel, last=300)
    if not e and tp:
        df = pd.DataFrame(tp)
        fig = go.Figure()
        if "low_95" in df:
            fig.add_trace(go.Scatter(x=list(df["target_time"]) + list(df["target_time"][::-1]),
                                     y=list(df["high_95"]) + list(df["low_95"][::-1]),
                                     fill="toself", line=dict(width=0), opacity=0.2, name="95% band"))
        fig.add_trace(go.Scatter(x=df["target_time"], y=df["actual"], name="Actual"))
        fig.add_trace(go.Scatter(x=df["target_time"], y=df["predicted"], name="Predicted",
                                 line=dict(dash="dash")))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------- chatbot ----------
st.divider()
st.subheader("🤖 Ask the AI assistant")
st.caption("Examples: What will gold be after 24 hours? How much is 21K gold in Ramallah? "
           "What does the news say about the Fed? How is the dollar index moving? "
           "What is the copper price doing for silver?")

if "chat" not in st.session_state:
    st.session_state.chat = []
for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)

question = st.chat_input("Ask about gold, silver, or the news...")
if question:
    st.session_state.chat.append(("user", question))
    st.chat_message("user").write(question)
    try:
        r = requests.post(f"{API}/chat", json={"question": question}, timeout=60)
        if r.status_code == 200:
            reply = r.json()["answer"]
        else:
            reply = f"Sorry, the chatbot is not ready: {r.json().get('detail', r.text)}"
    except requests.exceptions.RequestException:
        reply = "Sorry, I cannot reach the backend right now."
    st.session_state.chat.append(("assistant", reply))
    st.chat_message("assistant").write(reply)
