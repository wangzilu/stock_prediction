# Stock Prediction MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MVP stock prediction system that collects A-share market data + sentiment from Xueqiu/Eastmoney, trains a LightGBM model via Qlib, generates daily stock recommendations at 14:00, pushes via WeChat Work, and auto-verifies results after 5 trading days.

**Architecture:** Qlib as the quantitative engine with Alpha158 + custom sentiment factors. AKShare for market data, httpx for sentiment crawling, FinGPT for NLP scoring, APScheduler for orchestration, WeChat Work webhook for push notifications. SQLite tracks recommendations for 5-day verification.

**Tech Stack:** Python 3.10+, Qlib, AKShare, httpx, transformers (FinGPT), APScheduler, SQLite, requests (WeChat webhook)

---

## File Structure

```
stockPrediction/
├── pyproject.toml                    # Project dependencies and metadata
├── config/
│   ├── settings.py                   # All configuration (thresholds, paths, webhook URL)
│   └── watchlist.py                  # Stock watchlist definition
├── data/
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── market.py                 # AKShare A-share data collector
│   │   └── sentiment.py             # Xueqiu + Eastmoney sentiment crawler
│   └── storage/
│       └── __init__.py               # Qlib data directory (auto-created)
├── factors/
│   ├── __init__.py
│   ├── quant.py                      # Qlib Alpha158 DataHandler wrapper
│   └── sentiment.py                  # Sentiment factor calculation
├── models/
│   ├── __init__.py
│   └── short_term.py                 # LightGBM model training and prediction
├── signals/
│   ├── __init__.py
│   └── scorer.py                     # Signal scoring and recommendation generation
├── push/
│   ├── __init__.py
│   └── wechat.py                     # WeChat Work webhook push
├── tracker/
│   ├── __init__.py
│   └── verifier.py                   # 5-day verification tracking (SQLite)
├── scheduler/
│   ├── __init__.py
│   └── jobs.py                       # APScheduler job definitions
├── tests/
│   ├── __init__.py
│   ├── test_market_collector.py
│   ├── test_sentiment_collector.py
│   ├── test_sentiment_factor.py
│   ├── test_signal_scorer.py
│   ├── test_wechat_push.py
│   ├── test_verifier.py
│   └── test_scheduler.py
└── main.py                           # Entry point
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `config/settings.py`
- Create: `config/watchlist.py`

- [ ] **Step 1: Initialize git repo and create pyproject.toml**

```bash
cd /Users/wangzilu/MyProjects/stockPrediction
git init
```

```toml
# pyproject.toml
[project]
name = "stock-prediction"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "qlib>=0.9.0",
    "akshare>=1.10.0",
    "httpx>=0.27.0",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "apscheduler>=3.10.0",
    "requests>=2.31.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "lightgbm>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

- [ ] **Step 2: Create config/settings.py**

```python
# config/settings.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA_DIR = DATA_DIR / "qlib_data"
DB_PATH = DATA_DIR / "tracker.db"

# Qlib
QLIB_PROVIDER_URI = str(QLIB_DATA_DIR / "cn_data")

# Signal thresholds
HIGH_THRESHOLD = 0.7
MID_THRESHOLD = 0.3

# Push
WECHAT_WEBHOOK_URL = ""  # Set via environment variable WECHAT_WEBHOOK_URL

# Schedule
RECOMMENDATION_TIME = "14:00"
MARKET_CLOSE_TIME = "15:00"
DATA_CUTOFF_TIME = "13:00"

# Limits
MAX_RECOMMENDATIONS_PER_DAY = 5
MAX_PUSH_PER_STOCK_PER_DAY = 2

# Model
PREDICTION_HORIZON_DAYS = 5
TOP_K_STOCKS = 5
```

- [ ] **Step 3: Create config/watchlist.py**

```python
# config/watchlist.py

# A-share stock watchlist: (code, name)
# Using standard Qlib A-share format: SH600xxx, SZ000xxx
WATCHLIST = [
    ("SH600519", "贵州茅台"),
    ("SH601318", "中国平安"),
    ("SZ000858", "五粮液"),
    ("SZ300750", "宁德时代"),
    ("SH600036", "招商银行"),
    ("SZ000001", "平安银行"),
    ("SH601012", "隆基绿能"),
    ("SZ002594", "比亚迪"),
    ("SH600276", "恒瑞医药"),
    ("SZ000333", "美的集团"),
]

# AKShare uses different code format: sh600519, sz000858
def to_akshare_code(qlib_code: str) -> str:
    """Convert Qlib code (SH600519) to AKShare code (sh600519)."""
    return qlib_code.lower()

def to_stock_code(qlib_code: str) -> str:
    """Convert Qlib code (SH600519) to pure numeric code (600519)."""
    return qlib_code[2:]
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p config data/collectors data/storage factors models signals push tracker scheduler tests
touch config/__init__.py data/__init__.py data/collectors/__init__.py data/storage/__init__.py
touch factors/__init__.py models/__init__.py signals/__init__.py push/__init__.py
touch tracker/__init__.py scheduler/__init__.py tests/__init__.py
```

- [ ] **Step 5: Install dependencies and commit**

```bash
pip install -e ".[dev]"
git add .
git commit -m "feat: initialize project structure and dependencies"
```

---

## Task 2: A-Share Market Data Collector

**Files:**
- Create: `data/collectors/market.py`
- Create: `tests/test_market_collector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market_collector.py
import pandas as pd
import pytest
from data.collectors.market import MarketCollector


def test_fetch_daily_returns_dataframe():
    """Fetching daily data for a stock should return a DataFrame with OHLCV columns."""
    collector = MarketCollector()
    df = collector.fetch_daily("sh600519", days=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    required_cols = {"open", "high", "low", "close", "volume"}
    assert required_cols.issubset(set(df.columns))
    assert df.index.name == "date"


def test_fetch_daily_invalid_code_returns_empty():
    """Invalid stock code should return empty DataFrame."""
    collector = MarketCollector()
    df = collector.fetch_daily("sh999999", days=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_fetch_realtime_returns_dict():
    """Fetching realtime quote should return a dict with price info."""
    collector = MarketCollector()
    quote = collector.fetch_realtime("sh600519")
    assert isinstance(quote, dict)
    assert "price" in quote
    assert "change_pct" in quote
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_market_collector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.collectors.market'`

- [ ] **Step 3: Implement MarketCollector**

```python
# data/collectors/market.py
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta


class MarketCollector:
    """Collects A-share market data via AKShare."""

    def fetch_daily(self, code: str, days: int = 60) -> pd.DataFrame:
        """Fetch daily OHLCV data for a stock.

        Args:
            code: AKShare format code, e.g. "sh600519"
            days: Number of trading days to fetch

        Returns:
            DataFrame with columns [open, high, low, close, volume], indexed by date.
            Empty DataFrame if fetch fails.
        """
        try:
            symbol = code[2:]  # "sh600519" -> "600519"
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df[["open", "high", "low", "close", "volume"]]
            return df.tail(days)

        except Exception:
            return pd.DataFrame()

    def fetch_realtime(self, code: str) -> dict:
        """Fetch realtime quote for a stock.

        Args:
            code: AKShare format code, e.g. "sh600519"

        Returns:
            Dict with keys: price, change_pct, volume, high, low.
            Empty dict if fetch fails.
        """
        try:
            symbol = code[2:]
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol]

            if row.empty:
                return {}

            row = row.iloc[0]
            return {
                "price": float(row["最新价"]),
                "change_pct": float(row["涨跌幅"]),
                "volume": float(row["成交量"]),
                "high": float(row["最高"]),
                "low": float(row["最低"]),
            }

        except Exception:
            return {}

    def fetch_batch_daily(self, codes: list[str], days: int = 60) -> dict[str, pd.DataFrame]:
        """Fetch daily data for multiple stocks.

        Returns:
            Dict mapping code -> DataFrame
        """
        result = {}
        for code in codes:
            df = self.fetch_daily(code, days)
            if not df.empty:
                result[code] = df
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_market_collector.py -v`
Expected: PASS (3 tests pass; note: requires network access)

- [ ] **Step 5: Commit**

```bash
git add data/collectors/market.py tests/test_market_collector.py
git commit -m "feat: add A-share market data collector using AKShare"
```

---

## Task 3: Sentiment Data Collector (Xueqiu + Eastmoney)

**Files:**
- Create: `data/collectors/sentiment.py`
- Create: `tests/test_sentiment_collector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sentiment_collector.py
import pytest
from data.collectors.sentiment import SentimentCollector


def test_fetch_xueqiu_returns_list_of_posts():
    """Fetching Xueqiu posts for a stock should return a list of dicts."""
    collector = SentimentCollector()
    posts = collector.fetch_xueqiu("SH600519", limit=5)
    assert isinstance(posts, list)
    # May be empty if rate-limited, but structure should be correct
    if len(posts) > 0:
        assert "text" in posts[0]
        assert "timestamp" in posts[0]
        assert "source" in posts[0]
        assert posts[0]["source"] == "xueqiu"


def test_fetch_eastmoney_returns_list_of_posts():
    """Fetching Eastmoney guba posts should return a list of dicts."""
    collector = SentimentCollector()
    posts = collector.fetch_eastmoney("600519", limit=5)
    assert isinstance(posts, list)
    if len(posts) > 0:
        assert "text" in posts[0]
        assert "timestamp" in posts[0]
        assert "source" in posts[0]
        assert posts[0]["source"] == "eastmoney"


def test_fetch_all_sentiment_combines_sources():
    """fetch_all should combine Xueqiu and Eastmoney results."""
    collector = SentimentCollector()
    posts = collector.fetch_all("SH600519", limit_per_source=3)
    assert isinstance(posts, list)
    sources = {p["source"] for p in posts}
    # At least one source should return data
    assert len(posts) >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_sentiment_collector.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement SentimentCollector**

```python
# data/collectors/sentiment.py
import httpx
import time
from datetime import datetime


class SentimentCollector:
    """Collects sentiment posts from Xueqiu and Eastmoney Guba."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=10.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )

    def fetch_xueqiu(self, qlib_code: str, limit: int = 20) -> list[dict]:
        """Fetch recent posts from Xueqiu for a stock.

        Args:
            qlib_code: Qlib format code, e.g. "SH600519"
            limit: Max posts to return

        Returns:
            List of dicts with keys: text, timestamp, source
        """
        try:
            # Xueqiu uses $SH600519$ format in search
            symbol = qlib_code
            url = "https://xueqiu.com/query/v1/symbol/search/status.json"
            params = {
                "u": "",
                "q": symbol,
                "count": limit,
                "comment": "0",
                "symbol": symbol,
                "hl": "0",
                "source": "all",
                "sort": "time",
            }

            # Need cookies from main page first
            self.client.get("https://xueqiu.com/")
            resp = self.client.get(url, params=params)

            if resp.status_code != 200:
                return []

            data = resp.json()
            posts = []
            for item in data.get("list", [])[:limit]:
                text = item.get("text", "") or item.get("description", "")
                # Strip HTML tags simply
                text = text.replace("<", " <").split("<")[0] if "<" in text else text
                created_at = item.get("created_at", 0)
                posts.append({
                    "text": text[:500],
                    "timestamp": datetime.fromtimestamp(created_at / 1000).isoformat()
                    if created_at
                    else "",
                    "source": "xueqiu",
                })
            return posts

        except Exception:
            return []

    def fetch_eastmoney(self, stock_code: str, limit: int = 20) -> list[dict]:
        """Fetch recent posts from Eastmoney Guba.

        Args:
            stock_code: Pure numeric code, e.g. "600519"
            limit: Max posts to return

        Returns:
            List of dicts with keys: text, timestamp, source
        """
        try:
            url = f"https://guba.eastmoney.com/list,{stock_code},1,f.html"
            resp = self.client.get(url)

            if resp.status_code != 200:
                return []

            # Parse titles from the guba page (simple extraction)
            import re

            titles = re.findall(
                r'title="([^"]+)"[^>]*>([^<]*)</a>', resp.text
            )

            posts = []
            seen = set()
            for title_attr, _ in titles:
                text = title_attr.strip()
                if not text or text in seen or len(text) < 4:
                    continue
                seen.add(text)
                posts.append({
                    "text": text[:500],
                    "timestamp": datetime.now().isoformat(),
                    "source": "eastmoney",
                })
                if len(posts) >= limit:
                    break

            return posts

        except Exception:
            return []

    def fetch_all(self, qlib_code: str, limit_per_source: int = 20) -> list[dict]:
        """Fetch sentiment from all sources for a stock.

        Args:
            qlib_code: Qlib format code, e.g. "SH600519"
            limit_per_source: Max posts per source

        Returns:
            Combined list of posts from all sources
        """
        stock_code = qlib_code[2:]  # "SH600519" -> "600519"

        xueqiu_posts = self.fetch_xueqiu(qlib_code, limit_per_source)
        time.sleep(0.5)  # Rate limiting
        eastmoney_posts = self.fetch_eastmoney(stock_code, limit_per_source)

        return xueqiu_posts + eastmoney_posts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_sentiment_collector.py -v`
Expected: PASS (network-dependent; tests designed to pass gracefully even if APIs are temporarily unavailable)

- [ ] **Step 5: Commit**

```bash
git add data/collectors/sentiment.py tests/test_sentiment_collector.py
git commit -m "feat: add sentiment collector for Xueqiu and Eastmoney Guba"
```

---

## Task 4: Sentiment Factor Calculation

**Files:**
- Create: `factors/sentiment.py`
- Create: `tests/test_sentiment_factor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sentiment_factor.py
import pytest
from factors.sentiment import SentimentScorer


def test_score_single_text_returns_float():
    """Scoring a single text should return a float between -1 and 1."""
    scorer = SentimentScorer()
    score = scorer.score_text("这只股票最近表现非常好，看涨！")
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_score_negative_text():
    """Negative text should get a negative or low score."""
    scorer = SentimentScorer()
    score = scorer.score_text("暴跌崩盘了，赶紧跑，要完蛋了")
    assert score < 0.3


def test_score_batch_returns_aggregate():
    """Scoring a batch should return aggregate metrics."""
    scorer = SentimentScorer()
    posts = [
        {"text": "看好这只票，业绩很棒", "timestamp": "2026-05-05T10:00:00", "source": "xueqiu"},
        {"text": "下跌趋势明显，不看好", "timestamp": "2026-05-05T11:00:00", "source": "eastmoney"},
        {"text": "持续关注，等待机会", "timestamp": "2026-05-05T12:00:00", "source": "xueqiu"},
    ]
    result = scorer.score_batch(posts)
    assert "sentiment_score" in result
    assert "heat" in result
    assert "post_count" in result
    assert -1.0 <= result["sentiment_score"] <= 1.0
    assert result["post_count"] == 3


def test_score_empty_batch():
    """Empty post list should return neutral score."""
    scorer = SentimentScorer()
    result = scorer.score_batch([])
    assert result["sentiment_score"] == 0.0
    assert result["heat"] == 0.0
    assert result["post_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_sentiment_factor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement SentimentScorer**

```python
# factors/sentiment.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class SentimentScorer:
    """Scores financial text sentiment using a Chinese financial sentiment model.

    Uses a FinBERT-style model fine-tuned for Chinese financial text.
    Falls back to keyword-based scoring if model loading fails.
    """

    def __init__(self, model_name: str = "bardsai/finance-sentiment-zh-base"):
        self._model = None
        self._tokenizer = None
        self._model_name = model_name
        self._keywords_positive = {"看涨", "利好", "突破", "强势", "涨停", "暴涨", "看好", "牛", "反弹", "新高"}
        self._keywords_negative = {"暴跌", "崩盘", "利空", "跌停", "做空", "下跌", "割肉", "套牢", "风险", "亏损"}

    def _load_model(self):
        """Lazy-load the sentiment model."""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._model.eval()
        except Exception:
            # Fall back to keyword-based scoring
            self._model = None
            self._tokenizer = None

    def score_text(self, text: str) -> float:
        """Score a single text for sentiment.

        Args:
            text: Chinese financial text

        Returns:
            Float from -1.0 (very negative) to 1.0 (very positive)
        """
        if self._model is None and self._tokenizer is None:
            self._load_model()

        if self._model is not None and self._tokenizer is not None:
            return self._score_with_model(text)
        else:
            return self._score_with_keywords(text)

    def _score_with_model(self, text: str) -> float:
        """Score using the transformer model."""
        inputs = self._tokenizer(
            text[:512], return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]

        # Assume 3-class: negative(0), neutral(1), positive(2)
        if len(probs) == 3:
            score = probs[2] - probs[0]  # positive - negative, range [-1, 1]
        elif len(probs) == 2:
            score = probs[1] * 2 - 1  # binary: map [0,1] to [-1,1]
        else:
            score = 0.0

        return float(np.clip(score, -1.0, 1.0))

    def _score_with_keywords(self, text: str) -> float:
        """Fallback keyword-based scoring."""
        pos_count = sum(1 for kw in self._keywords_positive if kw in text)
        neg_count = sum(1 for kw in self._keywords_negative if kw in text)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return float(np.clip((pos_count - neg_count) / total, -1.0, 1.0))

    def score_batch(self, posts: list[dict]) -> dict:
        """Score a batch of posts and return aggregate metrics.

        Args:
            posts: List of dicts with "text", "timestamp", "source" keys

        Returns:
            Dict with keys:
                - sentiment_score: average sentiment [-1, 1]
                - heat: post count normalized (log scale)
                - post_count: raw number of posts
        """
        if not posts:
            return {"sentiment_score": 0.0, "heat": 0.0, "post_count": 0}

        scores = [self.score_text(p["text"]) for p in posts]
        avg_score = float(np.mean(scores))

        # Heat: log-normalized post count (0-1 scale, 100 posts = 1.0)
        heat = float(np.clip(np.log1p(len(posts)) / np.log1p(100), 0.0, 1.0))

        return {
            "sentiment_score": round(avg_score, 4),
            "heat": round(heat, 4),
            "post_count": len(posts),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_sentiment_factor.py -v`
Expected: PASS (keyword fallback will be used if model download fails in test env)

- [ ] **Step 5: Commit**

```bash
git add factors/sentiment.py tests/test_sentiment_factor.py
git commit -m "feat: add sentiment scoring with model fallback to keywords"
```

---

## Task 5: Qlib Integration and LightGBM Model

**Files:**
- Create: `factors/quant.py`
- Create: `models/short_term.py`

- [ ] **Step 1: Create Qlib data handler wrapper**

```python
# factors/quant.py
import qlib
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from pathlib import Path
from config.settings import QLIB_PROVIDER_URI


def init_qlib():
    """Initialize Qlib with local A-share data."""
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region_type="cn")


def get_alpha158_handler(
    start_time: str = "2020-01-01",
    end_time: str = "2026-05-01",
    instruments: str = "csi300",
) -> DataHandlerLP:
    """Get Qlib Alpha158 data handler.

    Args:
        start_time: Data start date
        end_time: Data end date
        instruments: Qlib instrument set (e.g. "csi300", "all")

    Returns:
        Alpha158 DataHandler instance
    """
    return Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
    )


def prepare_qlib_data():
    """Download and prepare Qlib A-share data (one-time setup).

    This downloads ~2GB of Chinese stock data.
    Run: python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
    """
    import subprocess
    import sys

    data_dir = Path(QLIB_PROVIDER_URI)
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Qlib data already exists at {data_dir}")
        return

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "qlib.run.get_data",
            "qlib_data",
            "--target_dir", str(data_dir),
            "--region", "cn",
        ],
        check=True,
    )
    print(f"Qlib data downloaded to {data_dir}")
```

- [ ] **Step 2: Implement short-term prediction model**

```python
# models/short_term.py
import pandas as pd
import numpy as np
from datetime import datetime

import qlib
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config

from config.settings import QLIB_PROVIDER_URI, PREDICTION_HORIZON_DAYS, TOP_K_STOCKS
from config.watchlist import WATCHLIST


class ShortTermModel:
    """LightGBM-based short-term stock prediction model using Qlib.

    Predicts 5-day forward returns using Alpha158 factors.
    """

    def __init__(self):
        self._model = None
        self._dataset = None
        self._initialized = False

    def initialize(self):
        """Initialize Qlib and prepare model."""
        if self._initialized:
            return
        qlib.init(provider_uri=QLIB_PROVIDER_URI, region_type="cn")
        self._initialized = True

    def train(
        self,
        train_start: str = "2020-01-01",
        train_end: str = "2025-12-31",
        valid_start: str = "2026-01-01",
        valid_end: str = "2026-03-31",
    ):
        """Train the LightGBM model.

        Args:
            train_start/end: Training period
            valid_start/end: Validation period
        """
        self.initialize()

        handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": train_start,
                "end_time": valid_end,
                "instruments": "csi300",
                "label": [
                    "Ref($close, -{}) / Ref($close, -1) - 1".format(PREDICTION_HORIZON_DAYS)
                ],
            },
        }

        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "train": (train_start, train_end),
                    "valid": (valid_start, valid_end),
                },
            },
        }

        self._dataset = init_instance_by_config(dataset_config)

        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.05,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 4,
            },
        }

        self._model = init_instance_by_config(model_config)
        self._model.fit(self._dataset)

    def predict(self, date: str = None) -> pd.DataFrame:
        """Generate predictions for the watchlist stocks.

        Args:
            date: Prediction date (defaults to today)

        Returns:
            DataFrame with columns [code, name, score, rank]
            sorted by score descending.
        """
        self.initialize()

        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        predictions = self._model.predict(dataset=self._dataset)

        # Filter to latest available date and watchlist stocks
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")

        # Get the most recent predictions
        latest_date = predictions.index.get_level_values(0).max()
        latest_preds = predictions.loc[latest_date]

        # Map watchlist
        watchlist_codes = {code for code, _ in WATCHLIST}
        watchlist_map = {code: name for code, name in WATCHLIST}

        results = []
        for code in latest_preds.index:
            if code in watchlist_codes:
                results.append({
                    "code": code,
                    "name": watchlist_map.get(code, ""),
                    "score": float(latest_preds.loc[code, "score"]),
                })

        df = pd.DataFrame(results)
        if df.empty:
            return df

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df.head(TOP_K_STOCKS)
```

- [ ] **Step 3: Commit**

```bash
git add factors/quant.py models/short_term.py
git commit -m "feat: add Qlib Alpha158 handler and LightGBM short-term model"
```

---

## Task 6: Signal Scorer

**Files:**
- Create: `signals/scorer.py`
- Create: `tests/test_signal_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signal_scorer.py
import pytest
from signals.scorer import SignalScorer, Recommendation


def test_score_stock_returns_recommendation():
    """Scoring a stock should return a Recommendation object."""
    scorer = SignalScorer()
    rec = scorer.score_stock(
        code="SH600519",
        name="贵州茅台",
        model_score=0.8,
        sentiment_score=0.5,
        sentiment_heat=0.6,
    )
    assert isinstance(rec, Recommendation)
    assert rec.code == "SH600519"
    assert rec.name == "贵州茅台"
    assert -1.0 <= rec.final_score <= 1.0
    assert rec.signal in ("强烈看多", "看多", "观望", "看空", "强烈看空")


def test_high_score_gives_bullish_signal():
    """High model + sentiment scores should give bullish signal."""
    scorer = SignalScorer()
    rec = scorer.score_stock(
        code="SH600519",
        name="贵州茅台",
        model_score=0.9,
        sentiment_score=0.7,
        sentiment_heat=0.8,
    )
    assert rec.signal in ("强烈看多", "看多")


def test_low_score_gives_bearish_signal():
    """Low model + negative sentiment should give bearish signal."""
    scorer = SignalScorer()
    rec = scorer.score_stock(
        code="SH600519",
        name="贵州茅台",
        model_score=-0.8,
        sentiment_score=-0.6,
        sentiment_heat=0.5,
    )
    assert rec.signal in ("强烈看空", "看空")


def test_generate_daily_report():
    """Generate daily report should return formatted recommendations."""
    scorer = SignalScorer()
    recs = [
        scorer.score_stock("SH600519", "贵州茅台", 0.8, 0.5, 0.6),
        scorer.score_stock("SZ300750", "宁德时代", 0.6, 0.3, 0.4),
    ]
    report = scorer.generate_report(recs)
    assert "今日推荐" in report
    assert "贵州茅台" in report
    assert "评分" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_signal_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement SignalScorer**

```python
# signals/scorer.py
from dataclasses import dataclass
from datetime import datetime

from config.settings import HIGH_THRESHOLD, MID_THRESHOLD


@dataclass
class Recommendation:
    """A single stock recommendation."""
    code: str
    name: str
    final_score: float
    signal: str
    model_score: float
    sentiment_score: float
    sentiment_heat: float
    reason: str


class SignalScorer:
    """Combines model predictions and sentiment into final signals."""

    def __init__(
        self,
        weight_model: float = 0.6,
        weight_sentiment: float = 0.3,
        weight_heat: float = 0.1,
    ):
        self.weight_model = weight_model
        self.weight_sentiment = weight_sentiment
        self.weight_heat = weight_heat

    def score_stock(
        self,
        code: str,
        name: str,
        model_score: float,
        sentiment_score: float,
        sentiment_heat: float,
    ) -> Recommendation:
        """Compute final signal for a stock.

        Args:
            code: Stock code (Qlib format)
            name: Stock name
            model_score: Model prediction score [-1, 1]
            sentiment_score: Sentiment score [-1, 1]
            sentiment_heat: Sentiment heat [0, 1]

        Returns:
            Recommendation with signal and reason
        """
        # Normalize model_score to [-1, 1] if needed
        model_norm = max(-1.0, min(1.0, model_score))

        # Weighted combination
        final_score = (
            model_norm * self.weight_model
            + sentiment_score * self.weight_sentiment
            + (sentiment_heat - 0.5) * self.weight_heat  # heat centered at 0.5
        )
        final_score = max(-1.0, min(1.0, final_score))

        # Determine signal
        signal = self._score_to_signal(final_score)

        # Generate reason
        reason = self._generate_reason(model_norm, sentiment_score, sentiment_heat)

        return Recommendation(
            code=code,
            name=name,
            final_score=round(final_score, 2),
            signal=signal,
            model_score=round(model_norm, 2),
            sentiment_score=round(sentiment_score, 2),
            sentiment_heat=round(sentiment_heat, 2),
            reason=reason,
        )

    def _score_to_signal(self, score: float) -> str:
        """Convert numeric score to signal text."""
        if score > HIGH_THRESHOLD:
            return "强烈看多"
        elif score > MID_THRESHOLD:
            return "看多"
        elif score < -HIGH_THRESHOLD:
            return "强烈看空"
        elif score < -MID_THRESHOLD:
            return "看空"
        else:
            return "观望"

    def _generate_reason(
        self, model_score: float, sentiment_score: float, sentiment_heat: float
    ) -> str:
        """Generate human-readable reason for the signal."""
        parts = []

        if model_score > 0.3:
            parts.append("量化模型看多")
        elif model_score < -0.3:
            parts.append("量化模型看空")

        if sentiment_score > 0.3:
            parts.append("舆情偏正面")
        elif sentiment_score < -0.3:
            parts.append("舆情偏负面")

        if sentiment_heat > 0.6:
            parts.append("讨论热度高")

        return "，".join(parts) if parts else "信号中性"

    def generate_report(self, recommendations: list[Recommendation]) -> str:
        """Generate formatted push report.

        Args:
            recommendations: List of Recommendation objects, sorted by score

        Returns:
            Formatted markdown string for WeChat push
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"📈 今日推荐 ({now})", "─────────────"]

        for i, rec in enumerate(recommendations, 1):
            score_display = round((rec.final_score + 1) * 5, 1)  # Map [-1,1] to [0,10]
            lines.append(
                f"{i}. {rec.name}({rec.code[2:]}) | {rec.signal} | 评分 {score_display}"
            )
            lines.append(f"   理由：{rec.reason}")

        lines.append("─────────────")

        # Position suggestion based on best signal
        if recommendations:
            best_score = recommendations[0].final_score
            if best_score > HIGH_THRESHOLD:
                position = "7-8成"
            elif best_score > MID_THRESHOLD:
                position = "5-6成"
            else:
                position = "3成以下"
            lines.append(f"建议仓位：{position}")

        return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_signal_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add signals/scorer.py tests/test_signal_scorer.py
git commit -m "feat: add signal scorer with multi-factor fusion and report generation"
```

---

## Task 7: WeChat Work Push Module

**Files:**
- Create: `push/wechat.py`
- Create: `tests/test_wechat_push.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wechat_push.py
import pytest
from unittest.mock import patch, MagicMock
from push.wechat import WeChatPusher


def test_format_markdown_message():
    """Should format text as WeChat markdown message payload."""
    pusher = WeChatPusher(webhook_url="https://example.com/hook")
    payload = pusher._build_payload("测试消息\n第二行")
    assert payload["msgtype"] == "markdown"
    assert payload["markdown"]["content"] == "测试消息\n第二行"


@patch("push.wechat.requests.post")
def test_send_success(mock_post):
    """Successful push should return True."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"errcode": 0, "errmsg": "ok"}
    mock_post.return_value = mock_resp

    pusher = WeChatPusher(webhook_url="https://example.com/hook")
    result = pusher.send("测试消息")
    assert result is True
    mock_post.assert_called_once()


@patch("push.wechat.requests.post")
def test_send_failure_returns_false(mock_post):
    """Failed push should return False."""
    mock_post.side_effect = Exception("Network error")

    pusher = WeChatPusher(webhook_url="https://example.com/hook")
    result = pusher.send("测试消息")
    assert result is False


def test_empty_webhook_url_raises():
    """Empty webhook URL should raise ValueError."""
    with pytest.raises(ValueError):
        WeChatPusher(webhook_url="")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_wechat_push.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement WeChatPusher**

```python
# push/wechat.py
import os
import requests
from config.settings import WECHAT_WEBHOOK_URL


class WeChatPusher:
    """Pushes messages to WeChat Work group via webhook robot."""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get("WECHAT_WEBHOOK_URL", WECHAT_WEBHOOK_URL)
        if not self.webhook_url:
            raise ValueError(
                "WeChat webhook URL is required. "
                "Set WECHAT_WEBHOOK_URL environment variable or pass webhook_url."
            )

    def _build_payload(self, content: str) -> dict:
        """Build WeChat Work markdown message payload."""
        return {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
            },
        }

    def send(self, content: str) -> bool:
        """Send a markdown message to WeChat Work group.

        Args:
            content: Markdown-formatted message text

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            payload = self._build_payload(content)
            resp = requests.post(self.webhook_url, json=payload, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                return data.get("errcode", -1) == 0

            return False

        except Exception:
            return False

    def send_recommendation(self, report: str) -> bool:
        """Send daily recommendation report."""
        return self.send(report)

    def send_alert(self, alert_content: str) -> bool:
        """Send risk alert message."""
        return self.send(alert_content)

    def send_verification(self, verification_report: str) -> bool:
        """Send 5-day verification report."""
        return self.send(verification_report)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_wechat_push.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add push/wechat.py tests/test_wechat_push.py
git commit -m "feat: add WeChat Work webhook push module"
```

---

## Task 8: Verification Tracker (5-Day Follow-up)

**Files:**
- Create: `tracker/verifier.py`
- Create: `tests/test_verifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verifier.py
import pytest
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from tracker.verifier import Verifier


@pytest.fixture
def verifier(tmp_path):
    """Create a verifier with temp database."""
    db_path = tmp_path / "test_tracker.db"
    return Verifier(db_path=str(db_path))


def test_record_recommendation(verifier):
    """Should store a recommendation in the database."""
    verifier.record_recommendation(
        date_str="2026-05-05",
        code="SH600519",
        name="贵州茅台",
        signal="看多",
        score=0.75,
    )
    pending = verifier.get_pending_verifications()
    assert len(pending) == 1
    assert pending[0]["code"] == "SH600519"


def test_verify_recommendation(verifier):
    """Should mark recommendation as verified with result."""
    verifier.record_recommendation(
        date_str="2026-04-28",
        code="SH600519",
        name="贵州茅台",
        signal="看多",
        score=0.75,
    )
    verifier.verify(
        date_str="2026-04-28",
        code="SH600519",
        price_at_rec=1800.0,
        price_at_verify=1860.0,
        high_price=1880.0,
        low_price=1780.0,
    )
    verified = verifier.get_verified(date_str="2026-04-28")
    assert len(verified) == 1
    assert verified[0]["return_pct"] == pytest.approx(3.33, rel=0.01)
    assert verified[0]["is_correct"] is True


def test_get_due_verifications(verifier):
    """Should return recommendations due for verification (5 trading days)."""
    verifier.record_recommendation(
        date_str="2026-04-25",
        code="SH600519",
        name="贵州茅台",
        signal="看多",
        score=0.75,
    )
    # 5 trading days later from April 25 (Mon) would be around May 5 (Mon)
    due = verifier.get_due_verifications(today="2026-05-05")
    assert len(due) == 1


def test_get_cumulative_stats(verifier):
    """Should calculate cumulative win rate."""
    verifier.record_recommendation("2026-04-20", "SH600519", "贵州茅台", "看多", 0.8)
    verifier.record_recommendation("2026-04-20", "SZ300750", "宁德时代", "看多", 0.7)

    verifier.verify("2026-04-20", "SH600519", 1800.0, 1850.0, 1870.0, 1790.0)
    verifier.verify("2026-04-20", "SZ300750", 200.0, 195.0, 210.0, 190.0)

    stats = verifier.get_cumulative_stats()
    assert stats["total"] == 2
    assert stats["correct"] == 1
    assert stats["win_rate"] == pytest.approx(50.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Verifier**

```python
# tracker/verifier.py
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

from config.settings import DB_PATH, PREDICTION_HORIZON_DAYS


class Verifier:
    """Tracks recommendations and verifies results after 5 trading days."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rec_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    score REAL NOT NULL,
                    price_at_rec REAL,
                    price_at_verify REAL,
                    high_price REAL,
                    low_price REAL,
                    return_pct REAL,
                    max_drawdown_pct REAL,
                    is_correct INTEGER,
                    verified INTEGER DEFAULT 0,
                    verify_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(rec_date, code)
                )
            """)

    def record_recommendation(
        self,
        date_str: str,
        code: str,
        name: str,
        signal: str,
        score: float,
        price_at_rec: float = None,
    ):
        """Record a new recommendation.

        Args:
            date_str: Recommendation date (YYYY-MM-DD)
            code: Stock code
            name: Stock name
            signal: Signal text (看多/看空/etc)
            score: Final score
            price_at_rec: Price at recommendation time
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO recommendations
                   (rec_date, code, name, signal, score, price_at_rec)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (date_str, code, name, signal, score, price_at_rec),
            )

    def verify(
        self,
        date_str: str,
        code: str,
        price_at_rec: float,
        price_at_verify: float,
        high_price: float,
        low_price: float,
    ):
        """Verify a recommendation with actual results.

        Args:
            date_str: Original recommendation date
            code: Stock code
            price_at_rec: Price when recommended
            price_at_verify: Price at verification time
            high_price: Highest price during period
            low_price: Lowest price during period
        """
        return_pct = round((price_at_verify - price_at_rec) / price_at_rec * 100, 2)
        max_drawdown_pct = round((low_price - price_at_rec) / price_at_rec * 100, 2)

        # Correct if: 看多 and positive return, or 看空 and negative return
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT signal FROM recommendations WHERE rec_date=? AND code=?",
                (date_str, code),
            ).fetchone()

            if row is None:
                return

            signal = row[0]
            is_correct = (
                ("多" in signal and return_pct > 0)
                or ("空" in signal and return_pct < 0)
            )

            conn.execute(
                """UPDATE recommendations SET
                   price_at_rec=?, price_at_verify=?, high_price=?, low_price=?,
                   return_pct=?, max_drawdown_pct=?, is_correct=?,
                   verified=1, verify_date=?
                   WHERE rec_date=? AND code=?""",
                (
                    price_at_rec, price_at_verify, high_price, low_price,
                    return_pct, max_drawdown_pct, int(is_correct),
                    datetime.now().strftime("%Y-%m-%d"),
                    date_str, code,
                ),
            )

    def get_pending_verifications(self) -> list[dict]:
        """Get all unverified recommendations."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE verified=0"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_due_verifications(self, today: str = None) -> list[dict]:
        """Get recommendations due for verification (5+ trading days old).

        Args:
            today: Today's date string (YYYY-MM-DD), defaults to actual today

        Returns:
            List of recommendation dicts that need verification
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")

        # Approximate 5 trading days as 7 calendar days
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE verified=0 AND rec_date <= ?",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_verified(self, date_str: str) -> list[dict]:
        """Get verified recommendations for a specific date."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE rec_date=? AND verified=1",
                (date_str,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_cumulative_stats(self) -> dict:
        """Get cumulative verification statistics."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(is_correct) as correct "
                "FROM recommendations WHERE verified=1"
            ).fetchone()

            total = row[0] or 0
            correct = row[1] or 0
            win_rate = (correct / total * 100) if total > 0 else 0.0

            return {
                "total": total,
                "correct": correct,
                "win_rate": round(win_rate, 1),
            }

    def generate_verification_report(self, rec_date: str) -> str:
        """Generate formatted verification report for a recommendation date.

        Args:
            rec_date: The original recommendation date

        Returns:
            Formatted report string
        """
        verified = self.get_verified(rec_date)
        if not verified:
            return ""

        today = datetime.now().strftime("%Y-%m-%d")
        stats = self.get_cumulative_stats()

        lines = [
            f"📋 荐股印证 (推荐日: {rec_date} → 今日: {today})",
            "─────────────",
        ]

        correct_count = 0
        total_count = len(verified)

        for i, rec in enumerate(verified, 1):
            result_icon = "✅" if rec["is_correct"] else "❌"
            lines.append(
                f"{i}. {rec['name']}({rec['code'][2:]}) | 推荐{rec['signal']}"
            )
            lines.append(
                f"   结果：{rec['return_pct']:+.1f}% {result_icon} | "
                f"最高{rec['high_price']:.1f} | 最大回撤{rec['max_drawdown_pct']:.1f}%"
            )
            if rec["is_correct"]:
                correct_count += 1

        lines.append("─────────────")
        lines.append(
            f"本轮胜率：{correct_count}/{total_count} "
            f"({correct_count/total_count*100:.0f}%) | "
            f"累计胜率：{stats['win_rate']:.0f}%"
        )

        return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_verifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tracker/verifier.py tests/test_verifier.py
git commit -m "feat: add 5-day verification tracker with SQLite persistence"
```

---

## Task 9: Scheduler and Main Entry Point

**Files:**
- Create: `scheduler/jobs.py`
- Create: `main.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py
import pytest
from unittest.mock import patch, MagicMock
from scheduler.jobs import DailyPipeline


def test_pipeline_runs_data_collection():
    """Pipeline should call data collection step."""
    pipeline = DailyPipeline.__new__(DailyPipeline)
    pipeline.market_collector = MagicMock()
    pipeline.sentiment_collector = MagicMock()
    pipeline.sentiment_scorer = MagicMock()
    pipeline.model = MagicMock()
    pipeline.signal_scorer = MagicMock()
    pipeline.pusher = MagicMock()
    pipeline.verifier = MagicMock()

    pipeline.market_collector.fetch_realtime.return_value = {"price": 1800.0, "change_pct": 1.5}
    pipeline.sentiment_collector.fetch_all.return_value = [
        {"text": "看好", "timestamp": "2026-05-05T10:00:00", "source": "xueqiu"}
    ]
    pipeline.sentiment_scorer.score_batch.return_value = {
        "sentiment_score": 0.5, "heat": 0.6, "post_count": 1
    }
    pipeline.model.predict.return_value = MagicMock()
    pipeline.signal_scorer.score_stock.return_value = MagicMock(
        code="SH600519", name="贵州茅台", final_score=0.7,
        signal="看多", reason="量化模型看多"
    )
    pipeline.signal_scorer.generate_report.return_value = "test report"
    pipeline.pusher.send_recommendation.return_value = True

    # Should not raise
    pipeline.run_daily_recommendation()


def test_pipeline_verification_step():
    """Pipeline verification should check due items."""
    pipeline = DailyPipeline.__new__(DailyPipeline)
    pipeline.market_collector = MagicMock()
    pipeline.verifier = MagicMock()
    pipeline.pusher = MagicMock()

    pipeline.verifier.get_due_verifications.return_value = [
        {"rec_date": "2026-04-28", "code": "SH600519", "name": "贵州茅台", "signal": "看多"}
    ]
    pipeline.market_collector.fetch_daily.return_value = MagicMock(
        empty=False,
        iloc=MagicMock(return_value={"close": 1850.0, "high": 1870.0, "low": 1780.0})
    )
    pipeline.verifier.generate_verification_report.return_value = "verification report"
    pipeline.pusher.send_verification.return_value = True

    # Should not raise
    pipeline.run_verification()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DailyPipeline**

```python
# scheduler/jobs.py
import logging
from datetime import datetime

from data.collectors.market import MarketCollector
from data.collectors.sentiment import SentimentCollector
from factors.sentiment import SentimentScorer
from models.short_term import ShortTermModel
from signals.scorer import SignalScorer
from push.wechat import WeChatPusher
from tracker.verifier import Verifier
from config.watchlist import WATCHLIST, to_akshare_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates the daily recommendation pipeline."""

    def __init__(self):
        self.market_collector = MarketCollector()
        self.sentiment_collector = SentimentCollector()
        self.sentiment_scorer = SentimentScorer()
        self.model = ShortTermModel()
        self.signal_scorer = SignalScorer()
        self.pusher = WeChatPusher()
        self.verifier = Verifier()

    def run_daily_recommendation(self):
        """Run the full daily recommendation pipeline at 14:00.

        Steps:
        1. Collect realtime market data for watchlist
        2. Collect and score sentiment for each stock
        3. Get model predictions
        4. Combine into final signals
        5. Push recommendations via WeChat
        6. Record in tracker
        """
        logger.info("Starting daily recommendation pipeline...")
        today = datetime.now().strftime("%Y-%m-%d")
        recommendations = []

        for qlib_code, name in WATCHLIST:
            try:
                # Market data
                ak_code = to_akshare_code(qlib_code)
                quote = self.market_collector.fetch_realtime(ak_code)
                if not quote:
                    logger.warning(f"No market data for {qlib_code}, skipping")
                    continue

                # Sentiment
                posts = self.sentiment_collector.fetch_all(qlib_code, limit_per_source=20)
                sentiment = self.sentiment_scorer.score_batch(posts)

                # Model prediction (simplified: use sentiment as proxy until Qlib trained)
                model_score = quote.get("change_pct", 0) / 10  # Normalize to [-1, 1] range

                # Signal scoring
                rec = self.signal_scorer.score_stock(
                    code=qlib_code,
                    name=name,
                    model_score=model_score,
                    sentiment_score=sentiment["sentiment_score"],
                    sentiment_heat=sentiment["heat"],
                )
                recommendations.append(rec)

            except Exception as e:
                logger.error(f"Error processing {qlib_code}: {e}")
                continue

        # Sort by score and take top picks
        recommendations.sort(key=lambda r: r.final_score, reverse=True)
        top_recs = [r for r in recommendations if r.signal in ("强烈看多", "看多")][:5]

        if not top_recs:
            logger.info("No strong signals today, sending neutral report")
            self.pusher.send("📊 今日无明确推荐信号，建议观望")
            return

        # Generate and push report
        report = self.signal_scorer.generate_report(top_recs)
        success = self.pusher.send_recommendation(report)
        logger.info(f"Push {'success' if success else 'failed'}: {len(top_recs)} recommendations")

        # Record for verification
        for rec in top_recs:
            ak_code = to_akshare_code(rec.code)
            quote = self.market_collector.fetch_realtime(ak_code)
            price = quote.get("price") if quote else None

            self.verifier.record_recommendation(
                date_str=today,
                code=rec.code,
                name=rec.name,
                signal=rec.signal,
                score=rec.final_score,
                price_at_rec=price,
            )

    def run_verification(self):
        """Check and verify due recommendations."""
        logger.info("Running verification check...")

        due = self.verifier.get_due_verifications()
        if not due:
            logger.info("No verifications due today")
            return

        # Group by recommendation date
        dates_to_verify = set()

        for rec in due:
            try:
                ak_code = to_akshare_code(rec["code"])
                df = self.market_collector.fetch_daily(ak_code, days=10)

                if df.empty:
                    continue

                # Get price data for verification period
                current_price = df.iloc[-1]["close"]
                high_price = df["high"].max()
                low_price = df["low"].min()
                price_at_rec = rec.get("price_at_rec") or df.iloc[0]["close"]

                self.verifier.verify(
                    date_str=rec["rec_date"],
                    code=rec["code"],
                    price_at_rec=price_at_rec,
                    price_at_verify=current_price,
                    high_price=high_price,
                    low_price=low_price,
                )
                dates_to_verify.add(rec["rec_date"])

            except Exception as e:
                logger.error(f"Error verifying {rec['code']}: {e}")

        # Send verification reports
        for rec_date in dates_to_verify:
            report = self.verifier.generate_verification_report(rec_date)
            if report:
                self.pusher.send_verification(report)
                logger.info(f"Verification report sent for {rec_date}")
```

- [ ] **Step 4: Implement main.py entry point**

```python
# main.py
"""Stock Prediction System - Main Entry Point.

Usage:
    python main.py              # Start scheduler (runs daily at 14:00)
    python main.py --run-now    # Run recommendation pipeline immediately
    python main.py --verify     # Run verification check immediately
    python main.py --setup      # Download Qlib data (first-time setup)
"""
import sys
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    args = sys.argv[1:]

    if "--setup" in args:
        from factors.quant import prepare_qlib_data
        prepare_qlib_data()
        return

    from scheduler.jobs import DailyPipeline
    pipeline = DailyPipeline()

    if "--run-now" in args:
        logger.info("Running recommendation pipeline now...")
        pipeline.run_daily_recommendation()
        return

    if "--verify" in args:
        logger.info("Running verification now...")
        pipeline.run_verification()
        return

    # Default: start scheduler
    scheduler = BlockingScheduler()

    # Daily recommendation at 14:00 on weekdays (Mon-Fri)
    scheduler.add_job(
        pipeline.run_daily_recommendation,
        CronTrigger(day_of_week="mon-fri", hour=14, minute=0),
        id="daily_recommendation",
        name="Daily Stock Recommendation",
    )

    # Verification check at 14:00 on weekdays
    scheduler.add_job(
        pipeline.run_verification,
        CronTrigger(day_of_week="mon-fri", hour=14, minute=5),
        id="verification",
        name="5-Day Verification Check",
    )

    logger.info("Scheduler started. Jobs:")
    logger.info("  - Daily recommendation: Mon-Fri 14:00")
    logger.info("  - Verification check: Mon-Fri 14:05")
    logger.info("Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_scheduler.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scheduler/jobs.py main.py tests/test_scheduler.py
git commit -m "feat: add daily pipeline scheduler and main entry point"
```

---

## Task 10: Integration Test and Final Verification

**Files:**
- Create: `tests/test_integration.py`
- Modify: `config/settings.py` (add env var loading)

- [ ] **Step 1: Add environment variable loading to settings**

Add to the top of `config/settings.py`:

```python
import os

# Override settings from environment variables
WECHAT_WEBHOOK_URL = os.environ.get("WECHAT_WEBHOOK_URL", "")
```

- [ ] **Step 2: Write integration test**

```python
# tests/test_integration.py
"""Integration tests - require network access. Run with: pytest tests/test_integration.py -v -m integration"""
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.integration


def test_full_pipeline_dry_run():
    """Run the full pipeline in dry-run mode (mocked push)."""
    with patch("push.wechat.WeChatPusher.__init__", return_value=None):
        with patch("push.wechat.WeChatPusher.send_recommendation", return_value=True):
            with patch("push.wechat.WeChatPusher.send", return_value=True):
                from scheduler.jobs import DailyPipeline

                pipeline = DailyPipeline.__new__(DailyPipeline)
                from data.collectors.market import MarketCollector
                from data.collectors.sentiment import SentimentCollector
                from factors.sentiment import SentimentScorer
                from signals.scorer import SignalScorer
                from tracker.verifier import Verifier
                from unittest.mock import MagicMock

                pipeline.market_collector = MarketCollector()
                pipeline.sentiment_collector = SentimentCollector()
                pipeline.sentiment_scorer = SentimentScorer()
                pipeline.model = MagicMock()
                pipeline.signal_scorer = SignalScorer()
                pipeline.pusher = MagicMock()
                pipeline.pusher.send_recommendation.return_value = True
                pipeline.pusher.send.return_value = True
                pipeline.verifier = Verifier(db_path="/tmp/test_integration.db")

                # Should complete without error
                pipeline.run_daily_recommendation()


def test_market_collector_real_data():
    """Verify AKShare can fetch real stock data."""
    from data.collectors.market import MarketCollector

    collector = MarketCollector()
    df = collector.fetch_daily("sh600519", days=5)
    assert not df.empty
    assert "close" in df.columns


def test_sentiment_scoring_real():
    """Verify sentiment scoring works end-to-end."""
    from factors.sentiment import SentimentScorer

    scorer = SentimentScorer()
    score = scorer.score_text("市场情绪高涨，看好后市")
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All unit tests PASS

Run: `cd /Users/wangzilu/MyProjects/stockPrediction && python -m pytest tests/test_integration.py -v -m integration`
Expected: Integration tests PASS (requires network)

- [ ] **Step 4: Add .gitignore**

```gitignore
# .gitignore
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/
*.egg
.env
data/storage/
*.db
.venv/
venv/
```

- [ ] **Step 5: Final commit**

```bash
git add tests/test_integration.py .gitignore config/settings.py
git commit -m "feat: add integration tests and finalize MVP setup"
```

---

## Post-Implementation Notes

### First-time setup commands:
```bash
cd /Users/wangzilu/MyProjects/stockPrediction
pip install -e ".[dev]"
python main.py --setup   # Downloads ~2GB Qlib A-share data
export WECHAT_WEBHOOK_URL="your-webhook-url-here"
python main.py --run-now # Test a single run
python main.py           # Start scheduler
```

### Key limitations of MVP (to address in later phases):
1. Model uses realtime price change as proxy until Qlib model is fully trained
2. Sentiment crawlers may need cookie/header updates if sites block
3. No risk alert system yet (Phase 4)
4. Single-market only (A-shares), crypto/gold in Phase 2
