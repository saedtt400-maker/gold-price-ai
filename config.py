"""Shared settings for the whole project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Metals. The high-accuracy version runs GOLD ONLY.
ALL_METALS = {
    "gold":   {"symbol": "XAU", "yahoo": "GC=F", "dukascopy": "XAUUSD", "name": "Gold"},
    "silver": {"symbol": "XAG", "yahoo": "SI=F", "dukascopy": "XAGUSD", "name": "Silver"},
}
ACTIVE_METALS = ["gold"]
METALS = {k: ALL_METALS[k] for k in ACTIVE_METALS}
USD_ILS_TICKER = "ILS=X"          # Yahoo Finance: how many shekels for 1 USD
GRAMS_PER_OUNCE = 31.1034768      # 1 troy ounce

# External API 1: live prices (free, no key)
LIVE_PRICE_URL = "https://api.gold-api.com/price/{symbol}"

# Real-time settings
POLL_SECONDS = 10                 # ask for a new live price every 10 seconds
HISTORY_REFRESH_SECONDS = 300     # refresh hourly history every 5 minutes
REQUEST_TIMEOUT = 10

# LSTM settings
SEQ_LENGTH = 48                   # look at the last 48 hours
HORIZONS = [1, 4, 8, 12, 24]      # predict the price this many hours ahead
TRAIN_START_HOURLY = "2003-05-01"  # Dukascopy hourly gold (free) starts around here
TRAIN_START_DAILY = "2000-01-01"   # daily drivers (Yahoo, FRED) from 2000
LIVE_LOOKBACK_DAYS = 120           # how much recent history the live system keeps
EPOCHS = 30
BATCH_SIZE = 512
INTERVAL_LEVELS = [0.80, 0.95, 0.99]   # confidence bands around each forecast

# Dukascopy free historical datafeed (hourly candles, one file per month)
DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m0:02d}/BID_candles_hour_1.bi5"
DUKASCOPY_MIN_URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m0:02d}/{d:02d}/BID_candles_min_1.bi5"
DUKASCOPY_WORKERS = 4
# price scale in the files + a sane price range used to auto-check the scale
DUKASCOPY_SYMBOLS = {
    "XAUUSD": {"divisor": 1000, "range": (200, 20000)},
    "XAGUSD": {"divisor": 1000, "range": (2, 500)},
    "EURUSD": {"divisor": 100000, "range": (0.5, 2.0)},
    "GBPUSD": {"divisor": 100000, "range": (0.9, 2.5)},
    "USDJPY": {"divisor": 1000, "range": (50, 250)},
    "USDCAD": {"divisor": 100000, "range": (0.8, 2.0)},
    "USDSEK": {"divisor": 100000, "range": (4, 15)},
    "USDCHF": {"divisor": 100000, "range": (0.5, 2.0)},
}
YAHOO_FX = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
            "USDCAD": "CAD=X", "USDSEK": "SEK=X", "USDCHF": "CHF=X",
            "XAUUSD": "GC=F", "XAGUSD": "SI=F"}
CACHE_DIR = BASE_DIR / "data" / "cache"

# Paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LIVE_LOG_CSV = DATA_DIR / "live_log.csv"


def model_path(metal):
    return MODELS_DIR / f"{metal}_lstm.keras"


def scaler_path(metal):
    return MODELS_DIR / f"{metal}_scaler.joblib"


def history_csv(metal):
    return DATA_DIR / f"{metal}_hourly.csv"


def test_pred_csv(metal):
    return RESULTS_DIR / f"{metal}_test_predictions.csv"


METRICS_JSON = RESULTS_DIR / "metrics.json"

# FastAPI address used by Streamlit
BACKEND_URL = "http://127.0.0.1:8000"

# Transformer chatbot (Hugging Face)
CHAT_MODEL = "google/flan-t5-base"

# Local market (Ramallah / West Bank): purity of each karat
GOLD_KARATS = {"24K": 24 / 24, "22K": 22 / 24, "21K": 21 / 24, "18K": 18 / 24, "14K": 14 / 24}
SILVER_PURITY = {"999": 0.999, "925": 0.925}
LOCAL_MARKET = "Ramallah, West Bank"

# ---------- News layer ----------
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
NEWS_REFRESH_SECONDS = 900        # every 15 minutes
NEWS_PER_TOPIC = 15
SENTIMENT_MODEL = "ProsusAI/finbert"

# topic -> (search query, weight in the final score, how the news affects gold)
#   "risk":   bad world news pushes gold UP (safe haven)
#   "policy": rate cuts push gold UP, rate hikes push gold DOWN
#   "direct": news about gold itself, positive = UP
NEWS_TOPICS = {
    "central_banks": {
        "label": "Central banks (Fed, ECB)",
        "query": '("federal reserve" OR "interest rates" OR "central bank" OR ECB)',
        "weight": 0.30, "effect": "policy"},
    "war_peace": {
        "label": "War and peace",
        "query": '(war OR ceasefire OR "peace talks" OR sanctions OR military)',
        "weight": 0.20, "effect": "risk"},
    "us_china": {
        "label": "USA and China",
        "query": '("US China" OR "China tariffs" OR "trade war" OR "Beijing Washington")',
        "weight": 0.15, "effect": "risk"},
    "middle_east": {
        "label": "Middle East",
        "query": '("Middle East" OR Iran OR Israel OR Gaza OR "Red Sea")',
        "weight": 0.20, "effect": "risk"},
    "gold_market": {
        "label": "Gold and silver market",
        "query": '("gold price" OR "silver price" OR bullion)',
        "weight": 0.15, "effect": "direct"},
}
DOVISH_WORDS = ["rate cut", "cuts rates", "cut rates", "lower rates", "easing", "dovish", "pause"]
HAWKISH_WORDS = ["rate hike", "raise rates", "raises rates", "hawkish", "tightening", "higher for longer"]
NEWS_SILVER_FACTOR = 0.8          # silver reacts a bit less than gold to safe-haven news
NEWS_ADJUST_STRENGTH = 0.5        # how strongly the news score shifts the LSTM forecast

# ---------- Market drivers (external factors) ----------
# source:
#   dk_fx     hourly FX from Dukascopy (since 2003) + Yahoo for the last days
#   dxy_synth hourly US Dollar Index rebuilt from 6 currency pairs (ICE formula)
#   yahoo_d   daily Yahoo data since 2000 (value known after the day ends)
#   fred      FRED CSV (daily / monthly), cot (CFTC weekly), gpr (daily)
# lag_days: days after the data date when the value is really public
FACTORS = {
    # Currency (hourly)
    "dxy":     dict(label="US Dollar Index (rebuilt hourly)", category="Currency", source="dxy_synth",
                    id="DXY", kind="hourly_price", metals=["gold", "silver"], relation="Usually inverse",
                    freq="Hourly"),
    "eurusd":  dict(label="EUR/USD", category="Currency", source="dk_fx", id="EURUSD",
                    kind="hourly_price", metals=["gold"], relation="Often positive", freq="Hourly"),
    "usdjpy":  dict(label="USD/JPY", category="Currency", source="dk_fx", id="USDJPY",
                    kind="hourly_price", metals=["gold"], relation="Often inverse", freq="Hourly"),
    # Interest rates
    "us10y":   dict(label="US 10Y Treasury yield", category="Interest rates", source="yahoo_d", id="^TNX",
                    kind="daily_rate", lag_days=0, metals=["gold", "silver"], relation="Usually inverse",
                    freq="Daily"),
    "real10y": dict(label="US 10Y real yield (TIPS)", category="Interest rates", source="fred", id="DFII10",
                    kind="daily_rate", lag_days=1, metals=["gold", "silver"], relation="Strong inverse",
                    freq="Daily (from 2003)"),
    "fedfunds": dict(label="Fed Funds Rate", category="Interest rates", source="fred", id="DFF",
                     kind="daily_rate", lag_days=1, metals=["gold", "silver"], relation="Usually inverse",
                     freq="Daily"),
    # Inflation
    "cpi":     dict(label="US CPI (YoY %)", category="Inflation", source="fred", id="CPIAUCSL",
                    kind="yoy", lag_days=45, metals=["gold", "silver"], relation="Can support", freq="Monthly"),
    "core_cpi": dict(label="US Core CPI (YoY %)", category="Inflation", source="fred", id="CPILFESL",
                     kind="yoy", lag_days=45, metals=["gold"], relation="Can support", freq="Monthly"),
    "pce":     dict(label="US PCE prices (YoY %)", category="Inflation", source="fred", id="PCEPI",
                    kind="yoy", lag_days=60, metals=["gold"], relation="Can support", freq="Monthly"),
    "breakeven": dict(label="10Y inflation expectations", category="Inflation", source="fred", id="T10YIE",
                      kind="daily_rate", lag_days=1, metals=["gold"], relation="Can support",
                      freq="Daily (from 2003)"),
    # Risk
    "gpr":     dict(label="Geopolitical Risk Index", category="Risk", source="gpr", id="GPRD",
                    kind="loglevel", lag_days=1, metals=["gold"], relation="Often positive", freq="Daily"),
    "epu":     dict(label="Economic Policy Uncertainty", category="Risk", source="fred", id="USEPUINDXD",
                    kind="loglevel", lag_days=1, metals=["gold"], relation="Often positive", freq="Daily"),
    "vix":     dict(label="VIX", category="Market risk", source="yahoo_d", id="^VIX",
                    kind="daily_rate", lag_days=0, metals=["gold", "silver"], relation="Often positive",
                    freq="Daily"),
    # Commodities
    "wti":     dict(label="WTI oil", category="Commodities", source="yahoo_d", id="CL=F",
                    kind="daily_price", lag_days=0, metals=["gold", "silver"],
                    relation="Supports via inflation", freq="Daily"),
    "brent":   dict(label="Brent oil", category="Commodities", source="yahoo_d", id="BZ=F",
                    kind="daily_price", lag_days=0, metals=["gold", "silver"],
                    relation="Supports via inflation", freq="Daily (from 2007)"),
    # Stocks
    "spx":     dict(label="S&P 500", category="Stocks", source="yahoo_d", id="^GSPC",
                    kind="daily_price", lag_days=0, metals=["gold"], relation="Varies", freq="Daily"),
    "nasdaq":  dict(label="Nasdaq", category="Stocks", source="yahoo_d", id="^IXIC",
                    kind="daily_price", lag_days=0, metals=["gold"], relation="Varies", freq="Daily"),
    # Investment demand
    "gld_flow": dict(label="Gold ETF activity (GLD volume)", category="Investment", source="yahoo_d",
                     id="GLD", kind="daily_volume", lag_days=0, metals=["gold"], relation="Important",
                     freq="Daily (from 2004)"),
    # Futures positioning
    "cot":     dict(label="Speculators net long (CFTC COT)", category="Futures", source="cot",
                    id="own_metal", kind="weekly_level", lag_days=3, metals=["gold", "silver"],
                    relation="Important", freq="Weekly"),
}

# Rebuilt US Dollar Index: DXY = 50.14348112 x EURUSD^-0.576 x USDJPY^0.136 x GBPUSD^-0.119
#                                 x USDCAD^0.091 x USDSEK^0.042 x USDCHF^0.036
DXY_WEIGHTS = {"EURUSD": -0.576, "USDJPY": 0.136, "GBPUSD": -0.119,
               "USDCAD": 0.091, "USDSEK": 0.042, "USDCHF": 0.036}
DXY_CONST = 50.14348112
FEATURES_REQUIRED_SHARE = 0.6    # a driver must cover 60% of training rows, else it is dropped

# Factors in the lists that have no free, frequent API. Shown in the dashboard as context only.
CONTEXT_ONLY = [
    ("Central-bank gold purchases / selling", "gold", "Quarterly (World Gold Council), no free API"),
    ("Physical gold demand", "gold", "Quarterly (World Gold Council)"),
    ("Gold mine production", "gold", "Annual / quarterly, changes slowly"),
    ("Gold recycling", "gold", "Quarterly, changes slowly"),
    ("ISM / global manufacturing PMI", "silver", "Monthly, licensed data (copper and industrial production used instead)"),
    ("Solar-panel silver demand", "silver", "Annual (Silver Institute); TAN ETF used as a live proxy"),
    ("Silver mine production", "silver", "Annual, changes slowly"),
    ("Silver recycling", "silver", "Annual, changes slowly"),
]

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_START = "1999-01-01"
CFTC_COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
COT_MARKETS = {"gold": "GOLD - COMMODITY EXCHANGE INC.", "silver": "SILVER - COMMODITY EXCHANGE INC."}
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
SLOW_FACTOR_REFRESH_SECONDS = 6 * 3600
DAILY_FACTOR_REFRESH_SECONDS = 30 * 60
FEATURE_SETS = ["base", "full"]     # trained and compared; the better one (validation) is used live


def factor_analysis_json(metal):
    return RESULTS_DIR / f"{metal}_factor_analysis.json"
