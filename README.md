# Real-Time AI Gold Price Prediction (long free history, market drivers, news)

A complete real-time AI application for **gold**. It trains on **more than 20 years of free hourly gold prices** (Dukascopy, from about 2003) and daily market drivers (from 2000), reads the live price every 10 seconds, and predicts the price **1, 4, 8, 12 and 24 hours ahead** with a confidence band (80%, 95%, 99%). It also reads world news with a Transformer, shows everything on a Streamlit dashboard in **shekels per gram (Ramallah market)**, and answers questions with a Transformer chatbot.

## 1. Project idea

Many families in Palestine keep their savings in gold. Local shops price gold in shekels per gram, while world markets price it in dollars per ounce and react to the dollar, interest rates, inflation, wars, and oil. This system joins the world price, the local price, the drivers, the news, and a short-term forecast with an honest measure of its uncertainty.

## 2. System architecture

```
 gold-api.com (live) ─┐   Dukascopy (hourly since ~2003)   FRED · CFTC · GPR · Yahoo daily (since 2000)
                      │              │                                   │
                      ▼              ▼                                   ▼
                 ┌──────────────────────── FastAPI backend ────────────────────────┐
                 │ price loop (10 s) · history (5 min) · drivers (5 min / 30 min / 6 h) │
                 └──────────────┬──────────────────────────────────────────────────┘
                                ▼
                 Data processing (returns, technicals, drivers placed at public time)
                                ▼
                 LSTM → 1/4/8/12/24 h forecast + 80/95/99% bands
                                ▲                         GDELT / Google News
                                └── news score (FinBERT) ◄────────┘
                                ▼
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
   Streamlit dashboard (10 s)              Transformer chatbot (Flan-T5)
```

## 3. Folder structure

```
metals_ai/
├── config.py                 all settings (gold only, horizons, sources, drivers)
├── requirements.txt
├── README.md
├── backend/
│   ├── price_client.py       live spot price (gold-api.com), Yahoo hourly helper, USD/ILS
│   ├── history_sources.py    Dukascopy downloader (with cache), rebuilt dollar index, Yahoo daily
│   ├── factors.py            market drivers: download, public-time placement, features
│   ├── processing.py         cleaning, technical indicators, feature sets
│   ├── predictor.py          loads the LSTM, forecasts + confidence bands
│   ├── news.py               headlines, FinBERT sentiment, news score
│   ├── chatbot.py            Flan-T5 chatbot
│   └── main.py               FastAPI app and background loops
├── model/train_lstm.py       download, train base and full models, evaluate, save
├── frontend/app.py           Streamlit dashboard
├── models/                   saved model and scaler
├── data/                     history CSV, live log, cache/ (Dukascopy months)
└── results/                  metrics.json, test predictions, factor analysis, charts
```

## 4. Data sources (all free, no API key)

| Source | What | Period | Used for |
|---|---|---|---|
| gold-api.com `GET /price/XAU` | Live gold spot, USD/oz | now | Price every 10 seconds |
| Dukascopy datafeed | Hourly candles: XAUUSD, EURUSD, USDJPY, GBPUSD, USDCAD, USDSEK, USDCHF | about 2003 to now | Training history and recent hours |
| Yahoo Finance (`yfinance`) | Newest hours not yet in Dukascopy; daily 10Y yield, VIX, WTI, Brent, S&P 500, Nasdaq, GLD volume; USD/ILS | daily since 2000 | Gap filling, daily drivers, shekel price |
| FRED CSV `fredgraph.csv?id=...` | Real yield, Fed Funds, CPI, Core CPI, PCE, inflation expectations, policy uncertainty | since 1999 | Macro drivers |
| CFTC Public Reporting API | Commitments of Traders (gold) | weekly | Speculator positioning |
| Caldara and Iacoviello | Daily Geopolitical Risk Index | daily | Risk driver |
| GDELT DOC 2.0 / Google News RSS | Headlines | last 24 h | News layer |

### 4.1 Dukascopy details

- URL: `https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YEAR}/{MONTH-1}/BID_candles_hour_1.bi5` (the month is counted from 0).
- Files are LZMA-compressed. Each record is 24 bytes: time offset, open, close, low, high (integers), volume (float).
- Prices are divided by a fixed scale (1000 for gold). The code checks that the result is in a sane range and tries other scales if not.
- Candles with zero volume (weekends, holidays) are removed.
- Finished months are cached in `data/cache/dukascopy/`, so the long download happens **once**. Recent months are refreshed every 15 minutes, and the newest days are built from 1-minute files.
- Dukascopy gives **spot** prices, the same kind as the live price, so history and live data match.
- Dukascopy data is offered for free; check their terms of use before any non-educational use.

### 4.2 Rebuilt hourly Dollar Index

The official DXY has no free long hourly history, so it is rebuilt from its six currencies with the ICE formula:

```
DXY = 50.14348112 × EURUSD^-0.576 × USDJPY^0.136 × GBPUSD^-0.119 × USDCAD^0.091 × USDSEK^0.042 × USDCHF^0.036
```

## 5. Market drivers

| Category | Driver | Frequency | Features | Expected relation |
|---|---|---|---|---|
| Currency | Dollar Index (rebuilt) | Hourly | 1h, 24h return | Usually inverse |
| Currency | EUR/USD | Hourly | 1h, 24h return | Often positive |
| Currency | USD/JPY | Hourly | 1h, 24h return | Often inverse |
| Interest rates | US 10Y yield | Daily | level, 5-day change | Usually inverse |
| Interest rates | US 10Y real yield (TIPS) | Daily, from 2003 | level, 5-day change | Strong inverse |
| Interest rates | Fed Funds Rate | Daily | level, 5-day change | Usually inverse |
| Inflation | CPI, Core CPI, PCE | Monthly | YoY % | Can support |
| Inflation | 10Y inflation expectations | Daily, from 2003 | level, 5-day change | Can support |
| Risk | Geopolitical Risk Index | Daily | log level | Often positive |
| Risk | Economic Policy Uncertainty | Daily | log level | Often positive |
| Market risk | VIX | Daily | level, 5-day change | Often positive |
| Commodities | WTI, Brent | Daily | 1-day, 5-day return | Support through inflation |
| Stocks | S&P 500, Nasdaq | Daily | 1-day, 5-day return | Varies |
| Investment | GLD volume (ETF flow proxy) | Daily, from 2004 | 60-day z-score | Important |
| Futures | CFTC speculators net long | Weekly | level, 4-week change | Important |
| Gold itself | price, returns, volatility, RSI(14), MACD, 24h and 120h averages | Hourly | base features | Very important |

A driver that covers less than 60% of the training hours is dropped (for example Brent, which starts in 2007). A driver that starts a bit later than 2003 is kept, and its missing early hours are filled with its training average.

**Not in the model** (shown as context): central-bank purchases, physical demand, mine production, recycling. They are quarterly or annual and have no free API.

### 5.1 No look-ahead (data leakage)

| Data | Used from |
|---|---|
| Hourly candles | end of the candle (start + 1 hour) |
| Yahoo daily values | next day 00:00 UTC |
| FRED daily | date + 2 days |
| CPI, Core CPI | month start + 46 days |
| PCE | month start + 61 days |
| CFTC COT (Tuesday) | + 4 days (published Friday) |
| GPR | date + 2 days |

Daily, weekly and monthly features (for example a 5-day change) are calculated at their own frequency first, then carried forward hour by hour.

## 6. LSTM model

**What it predicts:** the gold spot price **1, 4, 8, 12 and 24 trading hours** after now, in USD per ounce (then shown in shekels). One model gives all five answers.

| Item | Value |
|---|---|
| Input | last 48 hourly steps |
| Base features | log return, 4h and 24h momentum, distance from 24h and 120h averages, 24h volatility, RSI(14), MACD histogram, hour of day |
| Full features | base + drivers from section 5 |
| Targets | `ln(price_t+h / price_t)` for each horizon, each scaled by its own standard deviation |
| Layers | LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(64) → Dense(5) |
| Loss / optimizer | Huber / Adam, early stopping, learning-rate reduction |
| Split (by time) | 80% train, 10% validation, 10% test |
| Data pipeline | `tf.data` windows built on the fly, so 20+ years fit in memory |

Returns are used instead of raw prices because gold went from about $350 to above $4,000 over the period. Returns stay in a similar range across all years.

### 6.1 Base vs full (ablation)

Both feature sets are trained on the same dates. The one with the lower validation loss is used live. Both are reported. `results/gold_factor_analysis.json` holds each driver's correlation with the next 24-hour return and its permutation importance.

### 6.2 Confidence bands

After training, the model's errors on the **validation** period are measured for each horizon. The band that contains 80%, 95% and 99% of those errors becomes the live band:

```
low  = forecast × exp(−q_c,h)
high = forecast × exp(+q_c,h)
```

The bands are then checked on the **test** period ("band hit %"). If the 95% band holds the real price in about 95% of test cases, the band is reliable.

## 7. About accuracy (please read)

| Measure | What to expect | Why |
|---|---|---|
| Price accuracy (100 − MAPE), 1 hour | about 99.8% | Gold moves only about 0.1% to 0.3% in an hour |
| Naive model ("price stays the same"), 1 hour | also about 99.8% | Same reason |
| Direction accuracy (up or down) | around 50%, a few points above is already good | Short-term prices are close to a random walk |
| Band hit % | close to 80 / 95 / 99% | This is the reliable "high accuracy" number |

A test run of this code on **randomly generated prices** also gave 99.84% price accuracy at 1 hour, equal to the naive model, with about 50% direction accuracy. This shows that a very high price accuracy alone does not prove a model is intelligent. The fair way to judge the LSTM is:

1. its MAE compared with the naive MAE,
2. its direction accuracy compared with 50%,
3. how well its confidence bands hold on unseen data.

No free or paid data can make a 1 to 24 hour gold forecast 99.8% right about direction. More data and more drivers make the system more realistic and the bands more trustworthy, not certain.

## 8. Evaluation

### Results

Fill in after training (terminal output and `results/metrics.json`):

| Horizon | Features | Price accuracy % | Naive accuracy % | LSTM MAE $ | Naive MAE $ | Direction % | 95% band ±% | 95% band hit % |
|---|---|---|---|---|---|---|---|---|
| 1h | base | | | | | | | |
| 1h | full | | | | | | | |
| 4h | base | | | | | | | |
| 4h | full | | | | | | | |
| 8h | base | | | | | | | |
| 8h | full | | | | | | | |
| 12h | base | | | | | | | |
| 12h | full | | | | | | | |
| 24h | base | | | | | | | |
| 24h | full | | | | | | | |

Charts in `results/`: `gold_history.png` (full history, log scale) and `gold_actual_vs_predicted.png` (one panel per horizon with the 95% band).

## 9. News layer

### 9.1 Topics

| Topic | Examples in the search | Weight | How it affects gold |
|---|---|---|---|
| Central banks | Federal Reserve, interest rates, ECB | 30% | Rate cut → up, rate hike → down |
| War and peace | war, ceasefire, peace talks, sanctions | 20% | Bad world news → up (safe haven) |
| USA and China | trade war, tariffs, Beijing Washington | 15% | Tension → up |
| Middle East | Iran, Israel, Gaza, Red Sea | 20% | Tension → up, peace → down |
| Gold market | gold price, silver price, bullion | 15% | Positive news → up |

### 9.2 How the score is made

1. Download up to 15 headlines per topic from the last 24 hours.
2. **FinBERT** (`ProsusAI/finbert`, a Transformer for financial text) gives each headline a positive, negative, and neutral probability. Net sentiment = positive − negative.
3. Each headline gets a gold impact between −1 and +1, following the rule in the table above.
4. Topic score = average impact. Overall score = weighted average of topics.

| Overall score | Meaning |
|---|---|
| above +0.15 | News supports a higher gold price |
| −0.15 to +0.15 | Mixed or neutral |
| below −0.15 | News supports a lower gold price |

### 9.3 News-adjusted forecast

```
shift = 0.5 × news_score × hourly_volatility × √hours
news_adjusted_price = price_now × exp(LSTM_return + shift)
```

The shift grows with the horizon but stays inside normal market moves. **Honest note:** the news rule is based on known market logic, not learned from past data, because free news APIs give only recent headlines, not two years of history. For this reason the dashboard always shows the pure LSTM forecast and the news-adjusted forecast side by side, and only the pure LSTM is evaluated with metrics. Every live tick saves the news score in `data/live_log.csv`, so after some weeks this data can be used to test whether news really improves the forecast.

## 10. Transformer chatbot

- Model: `google/flan-t5-base` (encoder-decoder Transformer).
- For each question the backend builds short facts from the live state: current price in USD and ILS, 21K and 18K price in Ramallah, forecasts for every horizon (pure and with news), 24-hour statistics, the news summary, and top headlines per topic when the question is about news.
- Only the metal asked about is included, to stay inside the model's input size.
- If the model cannot load, a simple backup answer is used.

Example questions:
- What will the gold price be after 24 hours?
- Will silver go up in 4 hours?
- How much is 21K gold in Ramallah?
- What does the news say about the Federal Reserve?
- Is the Middle East news good or bad for gold?

## 11. Local market: Ramallah

| Metal | Purities | Formula |
|---|---|---|
| Gold | 24K, 22K, 21K, 18K, 14K | 24K gram price × karat / 24 |

For each purity the dashboard shows the price now, now plus workmanship (مصنعية, set in the sidebar), and the expected price after 1, 4, 8, 12 and 24 hours.

Check on 17 September 2026: 4,308.49 USD/oz and 13,078.05 ILS/oz give 24K = 420.45 ₪/g and 21K = 367.90 ₪/g, the same as the local market reference.

## 12. Error handling

| Situation | What happens |
|---|---|
| Live price API down | Last known price kept, warning shown |
| Dukascopy down or late | Yahoo hours fill the gap (scaled to match) |
| A driver source down | Its training average is used, warning shown with the driver name |
| GDELT down | Google News RSS used; if both fail, forecast uses LSTM only |
| FinBERT or Flan-T5 cannot load | Word list / simple answers used |
| Model files missing | Dashboard shows "run training first" |
| Any error inside a loop | Logged, loop continues next cycle |

## 13. Installation

Python 3.10 to 3.12.

```bash
cd metals_ai
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

## 14. How to run

**Step 1: train (once)**
```bash
python -m model.train_lstm
```
The first run downloads 20+ years of hourly data for gold and six currencies (a few thousand small files). This can take from several minutes to about an hour, depending on your connection. Training two models on about 130,000 hourly rows takes roughly 30 minutes to a few hours on a normal CPU, much less with a GPU. Later runs reuse the cache.

**Step 2: backend (terminal 1)**
```bash
uvicorn backend.main:app --port 8000
```

**Step 3: dashboard (terminal 2)**
```bash
streamlit run frontend/app.py
```

### Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, model, driver status |
| GET | `/latest` | live price, forecasts with bands, news summary, USD/ILS |
| GET | `/factors/gold` | drivers: latest value, change, correlation, importance, status |
| GET | `/news` | news topics and headlines |
| GET | `/ticks/gold` | prices collected every 10 s |
| GET | `/history/gold?hours=168` | hourly history |
| GET | `/comparison/gold?hours=72` | actual vs predicted (1 hour ahead) |
| GET | `/metrics` | test metrics |
| GET | `/test-predictions/gold?horizon=24` | test predictions with 95% band |
| POST | `/chat` | chatbot answer |

## 15. Notes and limits

- "24 hours" means 24 trading hours (weekend hours are not in the data).
- Shekel values use the current USD/ILS rate for all chart points.
- Silver can be turned back on in `config.ACTIVE_METALS`, but its driver list here is gold-focused.
- For learning purposes only. Not financial advice.
