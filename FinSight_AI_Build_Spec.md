# FinSight AI — Complete Project Build Specification

## Multi-Agent Equity Research Platform

**For use with Claude Code. Build each phase in order. Do not skip phases.**

> **Project:** FinSight AI — autonomous multi-agent equity research for Indian stocks  
> **Team:** Rajsekhar Singha Roy (23BAI1321) · Shubham Dubey (23BAI1344) · Ayush Achin (23BAI1055)  
> **Guide:** Khadar Nawas K | VIT Chennai | GenAI & Agentic AI Course Project

---

## TABLE OF CONTENTS

1. [Project Summary](#1-project-summary)  
2. [Tech Stack & Versions](#2-tech-stack--versions)  
3. [Complete Folder Structure](#3-complete-folder-structure)  
4. [Environment & Setup](#4-environment--setup)  
5. [Phase 1 — Data Pipeline](#5-phase-1--data-pipeline)  
6. [Phase 2 — RAG Pipeline](#6-phase-2--rag-pipeline)  
7. [Phase 3 — All Six Agents](#7-phase-3--all-six-agents)  
8. [Phase 4 — LangGraph Orchestration](#8-phase-4--langgraph-orchestration)  
9. [Phase 5 — FastAPI Backend](#9-phase-5--fastapi-backend)  
10. [Phase 6 — Gradio Frontend](#10-phase-6--gradio-frontend)  
11. [Phase 7 — Evaluation (RAGAS)](#11-phase-7--evaluation-ragas)  
12. [Static Data Files](#12-static-data-files)  
13. [Build Order for Claude Code](#13-build-order-for-claude-code)  
14. [Testing Checklist](#14-testing-checklist)

---

## 1\. Project Summary

### What it does

User types a stock ticker (e.g. `RELIANCE` or `HDFCBANK`). FinSight AI runs six specialized AI agents in a LangGraph-orchestrated pipeline and produces a structured investment research report covering:

- Fundamental analysis from actual company filing PDFs (via RAG)  
- Financial ratio analysis with sector benchmarking  
- News sentiment analysis with theme clustering  
- Technical price analysis (RSI, MACD, SMA)  
- India-specific risk scorecard (Beta, VaR, Altman Z-Score, Promoter Pledge)  
- Final synthesized report with Bull case / Bear case / Confidence score

### Key design principles

1. **Parallel execution** — four specialist agents run simultaneously via LangGraph `Send()` API  
2. **No hallucination** — financial ratios are computed in code, not inferred by the LLM  
3. **Auditability** — every claim traces to a specific agent and data source  
4. **India-first** — NSE/BSE filings, `.NS` yfinance tickers, promoter pledge flags, Nifty 50 Beta  
5. **Resilient** — non-fatal error handling: one agent failing does not stop the pipeline  
6. **Legal** — no scraping of nseindia.com or bseindia.com (Terms of Use restrict this)

### Data sources (legal & compliant)

| Source | What | API/Method |
| :---- | :---- | :---- |
| BSE/NSE company filings | Annual reports, quarterly results (PDFs) | Manual download \+ official BSE announcement API |
| yfinance | Price, OHLCV, fundamentals | `yfinance` Python library, `.NS` suffix |
| Google News RSS | Recent news headlines | `feedparser`, no API key needed |
| Screener.in | ROCE, CAGR, Promoter %, Pledge % | Pre-exported Excel, cached CSV |
| Nifty 50 | Market benchmark for Beta | yfinance `^NSEI` ticker |

---

## 2\. Tech Stack & Versions

\# Core LLM & Agent Framework

langchain==0.3.25

langchain-openai==0.2.14

langchain-community==0.3.24

langgraph==0.2.68

\# RAG & Document Processing

llama-index-core==0.12.6

llama-index-vector-stores-chroma==0.4.1

llama-index-embeddings-openai==0.3.1

llama-index-retrievers-bm25==0.5.0

llama-index-readers-file==0.4.4

\# Vector Databases

chromadb==0.6.3

faiss-cpu==1.9.0

\# Financial Data

yfinance==0.2.54

pandas==2.2.3

pandas-ta==0.3.14b0

numpy==1.26.4

\# News

feedparser==6.0.11

\# Backend

fastapi==0.115.12

uvicorn==0.34.0

pydantic==2.11.3

python-dotenv==1.1.0

tenacity==9.1.2

requests==2.32.3

\# Frontend

gradio==5.29.1

\# Evaluation

ragas==0.2.15

datasets==3.6.0

\# Export & Utilities

reportlab==4.4.1

openpyxl==3.1.5

pytest==8.3.5

httpx==0.28.1

### LLM Models

- **Specialist agents** (Filing Analyst, Ratio Cruncher, News Sentinel, Technical Analyst): `gpt-4o-mini`  
- **Report Writer** (final synthesis): `gpt-4o`  
- **Embeddings**: `text-embedding-3-small`

---

## 3\. Complete Folder Structure

finsight-ai/

│

├── .env                          \# API keys — never commit

├── .env.example                  \# Template for .env

├── .gitignore

├── requirements.txt

├── README.md

│

├── backend/

│   ├── \_\_init\_\_.py

│   │

│   ├── agents/

│   │   ├── \_\_init\_\_.py

│   │   ├── filing\_analyst.py     \# RAG over annual report PDFs

│   │   ├── ratio\_cruncher.py     \# LangChain ReAct \+ yfinance tools

│   │   ├── news\_sentinel.py      \# Google News RSS \+ LLM sentiment

│   │   ├── technical\_analyst.py  \# pandas\_ta indicators

│   │   ├── risk\_assessor.py      \# All risk metrics computation

│   │   └── report\_writer.py      \# Final synthesis with critique step

│   │

│   ├── graph/

│   │   ├── \_\_init\_\_.py

│   │   ├── state.py              \# FinSightState TypedDict definition

│   │   └── workflow.py           \# LangGraph StateGraph \+ Send() routing

│   │

│   ├── tools/

│   │   ├── \_\_init\_\_.py

│   │   ├── financial\_tools.py    \# yfinance wrapper @tool functions

│   │   ├── news\_tools.py         \# feedparser RSS fetch functions

│   │   └── technical\_tools.py   \# pandas\_ta indicator computation

│   │

│   ├── rag/

│   │   ├── \_\_init\_\_.py

│   │   ├── indexer.py            \# PDF → chunks → ChromaDB

│   │   └── retriever.py          \# Hybrid BM25 \+ vector retrieval

│   │

│   ├── risk/

│   │   ├── \_\_init\_\_.py

│   │   └── metrics.py            \# Beta, VaR, Altman Z-Score etc.

│   │

│   └── utils/

│       ├── \_\_init\_\_.py

│       ├── cache.py              \# JSON file-based caching (6hr TTL)

│       ├── sector\_data.py        \# Load sector\_medians.json

│       └── validators.py         \# Ticker format validation

│

├── main.py                       \# FastAPI app entry point

│

├── frontend/

│   └── app.py                    \# Gradio UI

│

├── data/

│   ├── filings/                  \# Raw PDFs organized by company

│   │   ├── RELIANCE/

│   │   │   ├── annual\_report\_FY24.pdf

│   │   │   └── q4\_results\_FY24.pdf

│   │   ├── HDFCBANK/

│   │   ├── INFY/

│   │   ├── TCS/

│   │   ├── WIPRO/

│   │   ├── BEL/

│   │   ├── TATAPOWER/

│   │   ├── BAJFINANCE/

│   │   ├── ADANIPORTS/

│   │   └── HINDUNILVR/

│   │

│   ├── chroma\_store/             \# Persisted ChromaDB (auto-created)

│   │

│   ├── screener\_cache/           \# Pre-downloaded Screener.in data

│   │   ├── RELIANCE.xlsx

│   │   ├── HDFCBANK.xlsx

│   │   └── ... (one per company)

│   │

│   └── cache/                    \# Runtime API response cache (JSON)

│

├── eval/

│   ├── test\_questions.json       \# 15 Q\&A pairs for RAGAS evaluation

│   └── ragas\_eval.py             \# RAGAS evaluation script

│

├── tests/

│   ├── test\_tools.py             \# Unit tests for financial\_tools.py

│   ├── test\_agents.py            \# Integration tests per agent

│   └── test\_graph.py             \# End-to-end graph test

│

└── static/

    └── sector\_medians.json       \# Hardcoded sector ratio medians

---

## 4\. Environment & Setup

### `.env` file

\# Required

OPENAI\_API\_KEY=sk-...

\# Optional (for RAGAS evaluation)

\# RAGAS does not require its own key — it uses OPENAI\_API\_KEY

\# App settings

APP\_HOST=0.0.0.0

APP\_PORT=8000

CACHE\_TTL\_HOURS=6

LOG\_LEVEL=INFO

### `.env.example`

OPENAI\_API\_KEY=your\_openai\_api\_key\_here

APP\_HOST=0.0.0.0

APP\_PORT=8000

CACHE\_TTL\_HOURS=6

LOG\_LEVEL=INFO

### `.gitignore`

.env

data/chroma\_store/

data/cache/

\_\_pycache\_\_/

\*.pyc

.DS\_Store

\*.egg-info/

dist/

build/

.pytest\_cache/

---

## 5\. Phase 1 — Data Pipeline

### 5.1 `backend/utils/cache.py` — Caching Layer

"""

Simple JSON file-based cache with TTL.

Prevents repeated API calls for the same ticker within TTL window.

"""

import json

import os

import time

import logging

from pathlib import Path

from typing import Any, Optional

logger \= logging.getLogger(\_\_name\_\_)

CACHE\_DIR \= Path("data/cache")

CACHE\_TTL\_HOURS \= int(os.getenv("CACHE\_TTL\_HOURS", "6"))

def \_cache\_path(key: str) \-\> Path:

    """Sanitize key and return cache file path."""

    safe\_key \= key.replace("/", "\_").replace(".", "\_").replace(" ", "\_")

    return CACHE\_DIR / f"{safe\_key}.json"

def cache\_get(key: str) \-\> Optional\[Any\]:

    """Return cached value if it exists and is not expired, else None."""

    path \= \_cache\_path(key)

    if not path.exists():

        return None

    try:

        with open(path) as f:

            data \= json.load(f)

        age\_hours \= (time.time() \- data\["\_ts"\]) / 3600

        if age\_hours \> CACHE\_TTL\_HOURS:

            logger.debug(f"Cache expired for {key}")

            return None

        logger.debug(f"Cache hit for {key}")

        return data\["value"\]

    except Exception as e:

        logger.warning(f"Cache read error for {key}: {e}")

        return None

def cache\_set(key: str, value: Any) \-\> None:

    """Write value to cache with current timestamp."""

    CACHE\_DIR.mkdir(parents=True, exist\_ok=True)

    path \= \_cache\_path(key)

    try:

        with open(path, "w") as f:

            json.dump({"\_ts": time.time(), "value": value}, f)

        logger.debug(f"Cached {key}")

    except Exception as e:

        logger.warning(f"Cache write error for {key}: {e}")

def cached(key\_prefix: str):

    """Decorator: cache the return value of a function."""

    def decorator(fn):

        def wrapper(\*args, \*\*kwargs):

            key \= f"{key\_prefix}\_{args\[0\] if args else ''}"

            result \= cache\_get(key)

            if result is not None:

                return result

            result \= fn(\*args, \*\*kwargs)

            cache\_set(key, result)

            return result

        return wrapper

    return decorator

### 5.2 `backend/tools/financial_tools.py` — yfinance Tools

"""

LangChain @tool wrappers around yfinance.

All functions handle .NS suffix automatically.

All returns are JSON-serializable dicts (no pandas objects).

"""

import logging

from typing import Optional

import yfinance as yf

import numpy as np

from langchain.tools import tool

from backend.utils.cache import cache\_get, cache\_set

logger \= logging.getLogger(\_\_name\_\_)

def \_ensure\_ns(ticker: str) \-\> str:

    """Add .NS suffix if not already present."""

    ticker \= ticker.strip().upper()

    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):

        return ticker \+ ".NS"

    return ticker

@tool

def get\_stock\_info(ticker: str) \-\> dict:

    """

    Fetch current stock info for an NSE-listed Indian company.

    ticker: NSE ticker symbol, e.g. 'RELIANCE' or 'RELIANCE.NS'

    Returns dict with price, market cap, P/E, P/B, ROE, D/E, beta, EPS, dividend yield.

    """

    ticker \= \_ensure\_ns(ticker)

    cache\_key \= f"stock\_info\_{ticker}"

    cached \= cache\_get(cache\_key)

    if cached:

        return cached

    try:

        stock \= yf.Ticker(ticker)

        info \= stock.info

        result \= {

            "ticker": ticker,

            "company\_name": info.get("longName", ""),

            "sector": info.get("sector", ""),

            "industry": info.get("industry", ""),

            "current\_price": info.get("currentPrice") or info.get("regularMarketPrice"),

            "market\_cap": info.get("marketCap"),

            "market\_cap\_cr": round(info.get("marketCap", 0\) / 1e7, 1\) if info.get("marketCap") else None,

            "pe\_ratio": info.get("trailingPE"),

            "forward\_pe": info.get("forwardPE"),

            "pb\_ratio": info.get("priceToBook"),

            "eps": info.get("trailingEps"),

            "dividend\_yield\_pct": round(info.get("dividendYield", 0\) \* 100, 2\) if info.get("dividendYield") else 0,

            "52w\_high": info.get("fiftyTwoWeekHigh"),

            "52w\_low": info.get("fiftyTwoWeekLow"),

            "beta": info.get("beta"),

            "roe\_pct": round(info.get("returnOnEquity", 0\) \* 100, 2\) if info.get("returnOnEquity") else None,

            "roa\_pct": round(info.get("returnOnAssets", 0\) \* 100, 2\) if info.get("returnOnAssets") else None,

            "debt\_to\_equity": info.get("debtToEquity"),

            "current\_ratio": info.get("currentRatio"),

            "revenue": info.get("totalRevenue"),

            "revenue\_cr": round(info.get("totalRevenue", 0\) / 1e7, 1\) if info.get("totalRevenue") else None,

            "net\_income": info.get("netIncomeToCommon"),

            "net\_income\_cr": round(info.get("netIncomeToCommon", 0\) / 1e7, 1\) if info.get("netIncomeToCommon") else None,

            "gross\_margins\_pct": round(info.get("grossMargins", 0\) \* 100, 2\) if info.get("grossMargins") else None,

            "operating\_margins\_pct": round(info.get("operatingMargins", 0\) \* 100, 2\) if info.get("operatingMargins") else None,

            "profit\_margins\_pct": round(info.get("profitMargins", 0\) \* 100, 2\) if info.get("profitMargins") else None,

            "shares\_outstanding": info.get("sharesOutstanding"),

            "book\_value": info.get("bookValue"),

            "free\_cashflow": info.get("freeCashflow"),

            "total\_debt": info.get("totalDebt"),

            "total\_cash": info.get("totalCash"),

        }

        cache\_set(cache\_key, result)

        return result

    except Exception as e:

        logger.error(f"yfinance get\_stock\_info error for {ticker}: {e}")

        return {"error": str(e), "ticker": ticker}

@tool

def get\_price\_history(ticker: str, period: str \= "1y") \-\> dict:

    """

    Fetch daily OHLCV price history for a stock.

    ticker: NSE ticker e.g. 'RELIANCE.NS'

    period: '1y', '2y', '6mo', '3mo'

    Returns dict with dates, closes, volumes as lists.

    """

    ticker \= \_ensure\_ns(ticker)

    cache\_key \= f"price\_history\_{ticker}\_{period}"

    cached \= cache\_get(cache\_key)

    if cached:

        return cached

    try:

        stock \= yf.Ticker(ticker)

        hist \= stock.history(period=period)

        if hist.empty:

            return {"error": f"No price data for {ticker}", "ticker": ticker}

        result \= {

            "ticker": ticker,

            "period": period,

            "dates": \[str(d.date()) for d in hist.index\],

            "open": \[round(v, 2\) for v in hist\["Open"\].tolist()\],

            "high": \[round(v, 2\) for v in hist\["High"\].tolist()\],

            "low": \[round(v, 2\) for v in hist\["Low"\].tolist()\],

            "close": \[round(v, 2\) for v in hist\["Close"\].tolist()\],

            "volume": hist\["Volume"\].tolist(),

            "latest\_price": round(hist\["Close"\].iloc\[-1\], 2),

            "oldest\_price": round(hist\["Close"\].iloc\[0\], 2),

            "price\_return\_pct": round(

                (hist\["Close"\].iloc\[-1\] / hist\["Close"\].iloc\[0\] \- 1\) \* 100, 2

            ),

        }

        cache\_set(cache\_key, result)

        return result

    except Exception as e:

        logger.error(f"yfinance get\_price\_history error for {ticker}: {e}")

        return {"error": str(e), "ticker": ticker}

@tool

def get\_nifty\_history(period: str \= "1y") \-\> dict:

    """

    Fetch Nifty 50 index price history for Beta computation.

    Returns dict with dates and closes as lists.

    """

    cache\_key \= f"nifty\_{period}"

    cached \= cache\_get(cache\_key)

    if cached:

        return cached

    try:

        nifty \= yf.Ticker("^NSEI")

        hist \= nifty.history(period=period)

        result \= {

            "dates": \[str(d.date()) for d in hist.index\],

            "close": \[round(v, 2\) for v in hist\["Close"\].tolist()\],

        }

        cache\_set(cache\_key, result)

        return result

    except Exception as e:

        logger.error(f"Nifty fetch error: {e}")

        return {"error": str(e)}

@tool

def get\_sector\_median(sector: str) \-\> dict:

    """

    Return median financial ratios for a given sector.

    sector: one of IT, BFSI, Auto, FMCG, Pharma, Infra, Defence, Energy, NBFC, Diversified

    """

    from backend.utils.sector\_data import get\_sector\_medians

    medians \= get\_sector\_medians()

    sector\_key \= sector.strip().upper()

    \# Try to match sector key

    for key in medians:

        if key.upper() \== sector\_key or sector\_key in key.upper():

            return {"sector": key, "medians": medians\[key\]}

    return {"sector": sector, "medians": medians.get("Diversified", {})}

### 5.3 `backend/tools/news_tools.py` — News Fetcher

"""

Google News RSS fetcher for Indian equity news.

No API key required. Uses feedparser.

"""

import logging

import feedparser

from datetime import datetime

from backend.utils.cache import cache\_get, cache\_set

logger \= logging.getLogger(\_\_name\_\_)

ECONOMIC\_TIMES\_MARKETS\_RSS \= "https://economictimes.indiatimes.com/markets/rss.cms"

ECONOMIC\_TIMES\_RESULTS\_RSS \= "https://economictimes.indiatimes.com/markets/earnings/rss.cms"

def fetch\_google\_news(company\_name: str, ticker: str \= "", limit: int \= 20\) \-\> list\[dict\]:

    """

    Fetch recent news for a company from Google News RSS.

    Returns list of article dicts with title, published, source, summary, link.

    """

    cache\_key \= f"news\_{company\_name.replace(' ', '\_')}"

    cached \= cache\_get(cache\_key)

    if cached:

        return cached

    \# Build query — company name \+ NSE/stock context for India filter

    query \= company\_name.replace(" ", "+")

    if ticker:

        query \+= f"+{ticker.replace('.NS', '')}"

    query \+= "+NSE+stock"

    url \= f"https://news.google.com/rss/search?q={query}\&hl=en-IN\&gl=IN\&ceid=IN:en"

    try:

        feed \= feedparser.parse(url)

        articles \= \[\]

        for entry in feed.entries\[:limit\]:

            articles.append({

                "title": entry.get("title", ""),

                "published": entry.get("published", ""),

                "source": entry.get("source", {}).get("title", "Google News"),

                "summary": entry.get("summary", ""),

                "link": entry.get("link", ""),

            })

        if not articles:

            logger.warning(f"No news found for {company\_name}, trying broader query")

            url2 \= f"https://news.google.com/rss/search?q={company\_name.replace(' ', '+')}\&hl=en-IN\&gl=IN\&ceid=IN:en"

            feed2 \= feedparser.parse(url2)

            for entry in feed2.entries\[:limit\]:

                articles.append({

                    "title": entry.get("title", ""),

                    "published": entry.get("published", ""),

                    "source": entry.get("source", {}).get("title", "Google News"),

                    "summary": entry.get("summary", ""),

                    "link": entry.get("link", ""),

                })

        cache\_set(cache\_key, articles)

        return articles

    except Exception as e:

        logger.error(f"News fetch error for {company\_name}: {e}")

        return \[\]

### 5.4 `backend/tools/technical_tools.py` — Technical Indicators

"""

Technical indicator computation using pandas\_ta.

All functions take OHLCV data as dict (from get\_price\_history) and return computed indicators.

"""

import logging

import pandas as pd

import pandas\_ta as ta

import numpy as np

logger \= logging.getLogger(\_\_name\_\_)

def compute\_technical\_indicators(price\_data: dict) \-\> dict:

    """

    Compute RSI, MACD, SMA-20, SMA-50 from OHLCV price dict.

    price\_data: output of get\_price\_history() tool

    Returns dict with all indicators and interpretations.

    """

    if "error" in price\_data or not price\_data.get("close"):

        return {"error": "No price data available"}

    try:

        df \= pd.DataFrame({

            "Open":   price\_data\["open"\],

            "High":   price\_data\["high"\],

            "Low":    price\_data\["low"\],

            "Close":  price\_data\["close"\],

            "Volume": price\_data\["volume"\],

        }, index=pd.to\_datetime(price\_data\["dates"\]))

        \# Compute indicators (appended as new columns)

        df.ta.rsi(length=14, append=True)

        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        df.ta.sma(length=20, append=True)

        df.ta.sma(length=50, append=True)

        df.ta.sma(length=200, append=True)

        latest \= df.iloc\[-1\]

        current\_price \= latest\["Close"\]

        \# Determine column names (pandas\_ta naming convention)

        rsi\_col    \= \[c for c in df.columns if c.startswith("RSI\_")\]

        macd\_col   \= \[c for c in df.columns if c.startswith("MACD\_") and not c.startswith("MACDh") and not c.startswith("MACDs")\]

        macd\_s\_col \= \[c for c in df.columns if c.startswith("MACDs\_")\]

        sma20\_col  \= \[c for c in df.columns if "SMA\_20" in c\]

        sma50\_col  \= \[c for c in df.columns if "SMA\_50" in c\]

        sma200\_col \= \[c for c in df.columns if "SMA\_200" in c\]

        rsi    \= round(latest\[rsi\_col\[0\]\], 2\)    if rsi\_col    and pd.notna(latest\[rsi\_col\[0\]\])    else None

        macd   \= round(latest\[macd\_col\[0\]\], 2\)   if macd\_col   and pd.notna(latest\[macd\_col\[0\]\])   else None

        macd\_s \= round(latest\[macd\_s\_col\[0\]\], 2\) if macd\_s\_col and pd.notna(latest\[macd\_s\_col\[0\]\]) else None

        sma20  \= round(latest\[sma20\_col\[0\]\], 2\)  if sma20\_col  and pd.notna(latest\[sma20\_col\[0\]\])  else None

        sma50  \= round(latest\[sma50\_col\[0\]\], 2\)  if sma50\_col  and pd.notna(latest\[sma50\_col\[0\]\])  else None

        sma200 \= round(latest\[sma200\_col\[0\]\], 2\) if sma200\_col and pd.notna(latest\[sma200\_col\[0\]\]) else None

        \# 52-week stats

        w52\_high \= max(price\_data\["high"\])

        w52\_low  \= min(price\_data\["low"\])

        w52\_pos  \= round((current\_price \- w52\_low) / (w52\_high \- w52\_low) \* 100, 1\) if (w52\_high \- w52\_low) \> 0 else 50

        \# Support / Resistance — rolling 20-day min/max from last 60 days

        recent \= df.tail(60)

        support    \= round(recent\["Low"\].rolling(20).min().iloc\[-1\], 2\)

        resistance \= round(recent\["High"\].rolling(20).max().iloc\[-1\], 2\)

        \# Interpretations

        trend \= "UPTREND" if sma50 and current\_price \> sma50 else "DOWNTREND"

        if sma200:

            if current\_price \> sma200:

                trend \+= " (above 200-day SMA — long-term bullish)"

            else:

                trend \+= " (below 200-day SMA — long-term bearish)"

        rsi\_label \= "OVERBOUGHT (\>70)" if rsi and rsi \> 70 else ("OVERSOLD (\<30)" if rsi and rsi \< 30 else "NEUTRAL (30–70)")

        macd\_label \= "BULLISH (MACD above signal)" if macd and macd\_s and macd \> macd\_s else "BEARISH (MACD below signal)"

        golden\_cross \= (sma20 and sma50 and sma20 \> sma50)

        death\_cross  \= (sma20 and sma50 and sma20 \< sma50)

        return {

            "current\_price": round(current\_price, 2),

            "trend": trend,

            "rsi": rsi,

            "rsi\_signal": rsi\_label,

            "macd": macd,

            "macd\_signal\_line": macd\_s,

            "macd\_direction": macd\_label,

            "sma\_20": sma20,

            "sma\_50": sma50,

            "sma\_200": sma200,

            "golden\_cross": golden\_cross,

            "death\_cross": death\_cross,

            "support\_level": support,

            "resistance\_level": resistance,

            "52w\_high": round(w52\_high, 2),

            "52w\_low": round(w52\_low, 2),

            "52w\_position\_pct": w52\_pos,

            "technicals\_summary": (

                f"{trend}. RSI {rsi} ({rsi\_label}). "

                f"MACD {macd\_label}. "

                f"Support ₹{support}, Resistance ₹{resistance}. "

                f"52-week position: {w52\_pos}%."

            ),

        }

    except Exception as e:

        logger.error(f"Technical indicator computation error: {e}")

        return {"error": str(e)}

### 5.5 `backend/utils/sector_data.py`

"""Load sector medians from static JSON file."""

import json

from pathlib import Path

\_SECTOR\_FILE \= Path("static/sector\_medians.json")

def get\_sector\_medians() \-\> dict:

    with open(\_SECTOR\_FILE) as f:

        return json.load(f)

def detect\_sector\_from\_yfinance(sector\_str: str) \-\> str:

    """Map yfinance sector string to our sector key."""

    mapping \= {

        "Technology": "IT",

        "Financial Services": "BFSI",

        "Consumer Defensive": "FMCG",

        "Healthcare": "Pharma",

        "Industrials": "Infra",

        "Energy": "Energy",

        "Basic Materials": "Energy",

        "Consumer Cyclical": "Auto",

        "Utilities": "Infra",

        "Communication Services": "IT",

        "Real Estate": "Infra",

    }

    return mapping.get(sector\_str, "Diversified")

---

## 6\. Phase 2 — RAG Pipeline

### 6.1 `backend/rag/indexer.py` — PDF Ingestion

"""

Ingest company annual report PDFs into ChromaDB via LlamaIndex.

Run once per company. Results persist to disk at data/chroma\_store/.

"""

import logging

from pathlib import Path

import chromadb

from llama\_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

from llama\_index.core.node\_parser import SentenceSplitter

from llama\_index.vector\_stores.chroma import ChromaVectorStore

from llama\_index.core import StorageContext

from llama\_index.embeddings.openai import OpenAIEmbedding

logger \= logging.getLogger(\_\_name\_\_)

CHROMA\_PATH \= Path("data/chroma\_store")

FILINGS\_PATH \= Path("data/filings")

\# Global embedding model — use small for cost efficiency

Settings.embed\_model \= OpenAIEmbedding(model="text-embedding-3-small")

def get\_chroma\_client() \-\> chromadb.PersistentClient:

    CHROMA\_PATH.mkdir(parents=True, exist\_ok=True)

    return chromadb.PersistentClient(path=str(CHROMA\_PATH))

def ingest\_company\_filings(company: str, force\_reindex: bool \= False) \-\> bool:

    """

    Index all PDFs in data/filings/{company}/ into ChromaDB.

    company: e.g. 'RELIANCE', 'HDFCBANK'

    Returns True if indexing succeeded, False otherwise.

    """

    pdf\_dir \= FILINGS\_PATH / company

    if not pdf\_dir.exists():

        logger.warning(f"No filing directory found for {company} at {pdf\_dir}")

        return False

    pdf\_files \= list(pdf\_dir.glob("\*.pdf"))

    if not pdf\_files:

        logger.warning(f"No PDFs found in {pdf\_dir}")

        return False

    client \= get\_chroma\_client()

    collection\_name \= f"filings\_{company.lower()}"

    \# Check if already indexed

    existing \= client.list\_collections()

    existing\_names \= \[c.name for c in existing\]

    if collection\_name in existing\_names and not force\_reindex:

        col \= client.get\_collection(collection\_name)

        if col.count() \> 0:

            logger.info(f"{company} already indexed ({col.count()} chunks). Skipping.")

            return True

    logger.info(f"Indexing {len(pdf\_files)} PDFs for {company}...")

    try:

        \# Load all PDFs

        documents \= SimpleDirectoryReader(

            input\_dir=str(pdf\_dir),

            required\_exts=\[".pdf"\],

        ).load\_data()

        \# Add company metadata to every document

        for doc in documents:

            doc.metadata\["company"\] \= company

            doc.metadata\["source\_type"\] \= "filing"

        \# Chunk: 512 tokens, 64-token overlap

        parser \= SentenceSplitter(chunk\_size=512, chunk\_overlap=64)

        nodes \= parser.get\_nodes\_from\_documents(documents)

        logger.info(f"Created {len(nodes)} chunks for {company}")

        \# Create/replace ChromaDB collection

        if collection\_name in existing\_names:

            client.delete\_collection(collection\_name)

        collection \= client.create\_collection(collection\_name)

        vector\_store \= ChromaVectorStore(chroma\_collection=collection)

        storage\_context \= StorageContext.from\_defaults(vector\_store=vector\_store)

        VectorStoreIndex(

            nodes=nodes,

            storage\_context=storage\_context,

        )

        logger.info(f"Successfully indexed {company}: {len(nodes)} chunks in ChromaDB")

        return True

    except Exception as e:

        logger.error(f"Indexing failed for {company}: {e}")

        return False

def ingest\_all\_companies(force\_reindex: bool \= False) \-\> dict:

    """Index all companies that have filing directories."""

    results \= {}

    for company\_dir in sorted(FILINGS\_PATH.iterdir()):

        if company\_dir.is\_dir():

            results\[company\_dir.name\] \= ingest\_company\_filings(

                company\_dir.name, force\_reindex=force\_reindex

            )

    return results

def company\_is\_indexed(company: str) \-\> bool:

    """Check if a company's filings are already in ChromaDB."""

    client \= get\_chroma\_client()

    collection\_name \= f"filings\_{company.lower()}"

    try:

        existing \= \[c.name for c in client.list\_collections()\]

        if collection\_name not in existing:

            return False

        col \= client.get\_collection(collection\_name)

        return col.count() \> 0

    except Exception:

        return False

### 6.2 `backend/rag/retriever.py` — Hybrid Retrieval

"""

Hybrid BM25 \+ dense vector retriever for company filing queries.

BM25 catches exact financial terms; dense catches semantic similarity.

Combined via Reciprocal Rank Fusion (RRF).

"""

import logging

import chromadb

from pathlib import Path

from llama\_index.core import VectorStoreIndex, Settings

from llama\_index.vector\_stores.chroma import ChromaVectorStore

from llama\_index.core import StorageContext

from llama\_index.retrievers.bm25 import BM25Retriever

from llama\_index.core.retrievers import QueryFusionRetriever

from llama\_index.embeddings.openai import OpenAIEmbedding

logger \= logging.getLogger(\_\_name\_\_)

CHROMA\_PATH \= Path("data/chroma\_store")

Settings.embed\_model \= OpenAIEmbedding(model="text-embedding-3-small")

def build\_hybrid\_retriever(company: str, top\_k: int \= 6):

    """

    Build a hybrid BM25 \+ vector retriever for a company's filings.

    Returns (retriever, index) tuple or (None, None) if company not indexed.

    """

    client \= chromadb.PersistentClient(path=str(CHROMA\_PATH))

    collection\_name \= f"filings\_{company.lower()}"

    try:

        collection \= client.get\_collection(collection\_name)

    except Exception:

        logger.error(f"Collection {collection\_name} not found in ChromaDB")

        return None, None

    vector\_store \= ChromaVectorStore(chroma\_collection=collection)

    storage\_context \= StorageContext.from\_defaults(vector\_store=vector\_store)

    index \= VectorStoreIndex.from\_vector\_store(

        vector\_store=vector\_store,

        storage\_context=storage\_context,

    )

    \# Dense vector retriever

    vector\_retriever \= index.as\_retriever(similarity\_top\_k=top\_k)

    \# BM25 keyword retriever

    bm25\_retriever \= BM25Retriever.from\_defaults(

        index=index,

        similarity\_top\_k=top\_k,

    )

    \# Fuse both using Reciprocal Rank Fusion

    hybrid\_retriever \= QueryFusionRetriever(

        retrievers=\[vector\_retriever, bm25\_retriever\],

        similarity\_top\_k=top\_k,

        num\_queries=1,              \# Don't generate sub-queries

        mode="reciprocal\_rerank",  \# RRF fusion

        use\_async=False,

    )

    return hybrid\_retriever, index

def retrieve\_for\_query(company: str, query: str, top\_k: int \= 6\) \-\> list\[str\]:

    """

    Retrieve top\_k relevant text chunks for a query from a company's filings.

    Returns list of text strings (the actual chunk content).

    """

    retriever, \_ \= build\_hybrid\_retriever(company, top\_k=top\_k)

    if retriever is None:

        return \[\]

    try:

        nodes \= retriever.retrieve(query)

        return \[node.get\_content() for node in nodes\]

    except Exception as e:

        logger.error(f"Retrieval error for {company} / {query}: {e}")

        return \[\]

---

## 7\. Phase 3 — All Six Agents

### 7.1 `backend/agents/filing_analyst.py`

"""

Filing Analyst Agent.

Uses hybrid RAG over pre-indexed annual report PDFs.

Extracts: revenue trend, margins, management themes, auditor flags, stated risks.

"""

import logging

import json

from langchain\_openai import ChatOpenAI

from backend.rag.retriever import retrieve\_for\_query

from backend.rag.indexer import company\_is\_indexed, ingest\_company\_filings

logger \= logging.getLogger(\_\_name\_\_)

llm \= ChatOpenAI(model="gpt-4o-mini", temperature=0)

\# Queries to run against the filing RAG index

FILING\_QUERIES \= \[

    "revenue sales income growth year financial performance",

    "EBITDA operating margin profitability gross margin",

    "management commentary outlook guidance growth strategy future",

    "risks challenges headwinds concerns mentioned risk factors",

    "debt borrowings interest coverage solvency capital structure",

    "cash flow operations working capital liquidity",

\]

def run\_filing\_analyst(ticker: str, company: str) \-\> dict:

    """

    Retrieve and analyze company filings using hybrid RAG.

    Returns structured dict with all extracted insights.

    """

    logger.info(f"\[FilingAnalyst\] Starting for {company} ({ticker})")

    \# Ensure company is indexed

    if not company\_is\_indexed(company):

        logger.info(f"\[FilingAnalyst\] Indexing {company} filings...")

        success \= ingest\_company\_filings(company)

        if not success:

            return {

                "status": "no\_filings",

                "message": f"No annual report PDFs found for {company}. Download PDFs to data/filings/{company}/",

                "company": company,

            }

    \# Run all queries and collect retrieved chunks

    all\_chunks \= \[\]

    for query in FILING\_QUERIES:

        chunks \= retrieve\_for\_query(company, query, top\_k=5)

        all\_chunks.extend(chunks)

    \# Deduplicate

    seen \= set()

    unique\_chunks \= \[\]

    for chunk in all\_chunks:

        if chunk\[:100\] not in seen:

            seen.add(chunk\[:100\])

            unique\_chunks.append(chunk)

    if not unique\_chunks:

        return {

            "status": "no\_content\_retrieved",

            "message": "RAG retrieval returned no results",

            "company": company,

        }

    context \= "\\n\\n---\\n\\n".join(unique\_chunks\[:20\])  \# Limit context size

    prompt \= f"""You are a senior equity research analyst reviewing company filings.

Company: {company}

The following are excerpts retrieved from the company's annual reports and financial filings:

{context}

Based ONLY on the above excerpts (do not use general knowledge), extract the following.

Return ONLY valid JSON. No preamble.

{{

  "revenue\_trend": "describe revenue growth with specific numbers if available",

  "ebitda\_margin": "EBITDA or operating margin with specific % if available",

  "net\_profit\_trend": "net profit or PAT trend with numbers",

  "management\_themes": \["list", "of", "key", "themes", "from", "management", "commentary"\],

  "growth\_drivers": \["key", "growth", "drivers", "mentioned"\],

  "auditor\_flags": \["any", "audit", "qualifications", "or", "emphasis", "of", "matter"\],

  "stated\_risks": \["key", "risks", "explicitly", "mentioned", "in", "filings"\],

  "debt\_situation": "description of debt/leverage situation",

  "cash\_flow\_status": "description of operating cash flow health",

  "key\_metrics\_mentioned": {{"metric\_name": "value\_from\_filing"}},

  "overall\_filing\_assessment": "one paragraph overall assessment based purely on filings"

}}"""

    try:

        response \= llm.invoke(prompt)

        content \= response.content.strip()

        if content.startswith("\`\`\`"):

            content \= content.split("\`\`\`")\[1\]

            if content.startswith("json"):

                content \= content\[4:\]

        result \= json.loads(content.strip())

        result\["status"\] \= "success"

        result\["company"\] \= company

        result\["chunks\_retrieved"\] \= len(unique\_chunks)

        logger.info(f"\[FilingAnalyst\] Complete for {company}: {len(unique\_chunks)} chunks")

        return result

    except json.JSONDecodeError as e:

        logger.error(f"\[FilingAnalyst\] JSON parse error: {e}")

        return {

            "status": "parse\_error",

            "raw\_response": response.content\[:500\],

            "company": company,

        }

    except Exception as e:

        logger.error(f"\[FilingAnalyst\] Error: {e}")

        return {"status": "error", "message": str(e), "company": company}

### 7.2 `backend/agents/ratio_cruncher.py`

"""

Ratio Cruncher Agent.

Uses LangChain ReAct loop with yfinance tools.

Computes 8+ financial ratios and benchmarks against sector medians.

"""

import logging

import json

from langchain\_openai import ChatOpenAI

from langchain.agents import AgentExecutor, create\_react\_agent

from langchain.prompts import PromptTemplate

from backend.tools.financial\_tools import get\_stock\_info, get\_sector\_median

logger \= logging.getLogger(\_\_name\_\_)

TOOLS \= \[get\_stock\_info, get\_sector\_median\]

REACT\_PROMPT \= PromptTemplate.from\_template("""

You are a financial analyst computing and benchmarking financial ratios for an Indian listed company.

Your task:

1\. Fetch stock info using get\_stock\_info tool

2\. Fetch sector medians using get\_sector\_median tool  

3\. Compute assessments by comparing each ratio to sector median

4\. Return structured JSON analysis

Company: {company}

Ticker: {ticker}

Sector: {sector}

Available tools: {tools}

Tool names: {tool\_names}

Use this format:

Thought: what I need to do

Action: tool\_name

Action Input: input\_value

Observation: tool\_result

... (repeat as needed)

Thought: I have all data needed

Final Answer: \[JSON response\]

{agent\_scratchpad}

IMPORTANT: Your Final Answer must be valid JSON in this exact format:

{{

  "ratios": \[

    {{"metric": "P/E Ratio", "value": 24.5, "sector\_median": 22.0, "unit": "x", "assessment": "Slightly above sector — marginal premium"}},

    {{"metric": "P/B Ratio", "value": 3.2, "sector\_median": 2.8, "unit": "x", "assessment": "Inline with sector"}},

    {{"metric": "ROE", "value": 14.2, "sector\_median": 12.0, "unit": "%", "assessment": "Above sector — good capital efficiency"}},

    {{"metric": "Debt/Equity", "value": 0.42, "sector\_median": 0.75, "unit": "x", "assessment": "Below sector — conservatively leveraged"}},

    {{"metric": "Operating Margin", "value": 17.5, "sector\_median": 14.0, "unit": "%", "assessment": "Above sector — operational efficiency"}},

    {{"metric": "Net Profit Margin", "value": 9.8, "sector\_median": 8.5, "unit": "%", "assessment": "Above sector — good profitability"}}

  \],

  "valuation\_verdict": "overall valuation assessment in one sentence",

  "financial\_health": "one sentence on balance sheet health",

  "current\_price": 2847.50,

  "market\_cap\_cr": 192400

}}

""")

def run\_ratio\_cruncher(ticker: str, company: str, sector: str) \-\> dict:

    """

    Run the ReAct agent to compute and benchmark financial ratios.

    Returns structured dict with all ratio assessments.

    """

    logger.info(f"\[RatioCruncher\] Starting for {company} ({ticker}) in {sector}")

    llm \= ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent \= create\_react\_agent(llm=llm, tools=TOOLS, prompt=REACT\_PROMPT)

    executor \= AgentExecutor(

        agent=agent,

        tools=TOOLS,

        verbose=False,

        max\_iterations=8,

        handle\_parsing\_errors=True,

    )

    try:

        result \= executor.invoke({

            "company": company,

            "ticker": ticker,

            "sector": sector,

        })

        output \= result.get("output", "")

        \# Parse JSON from output

        if "\`\`\`" in output:

            output \= output.split("\`\`\`")\[1\]

            if output.startswith("json"):

                output \= output\[4:\]

        \# Find JSON block

        start \= output.find("{")

        end \= output.rfind("}") \+ 1

        if start \>= 0 and end \> start:

            json\_str \= output\[start:end\]

            data \= json.loads(json\_str)

            data\["status"\] \= "success"

            data\["ticker"\] \= ticker

            data\["company"\] \= company

            logger.info(f"\[RatioCruncher\] Complete for {company}")

            return data

        return {"status": "parse\_error", "raw": output\[:300\], "company": company}

    except Exception as e:

        logger.error(f"\[RatioCruncher\] Error: {e}")

        \# Fallback: fetch data directly without ReAct loop

        return \_fallback\_ratio\_fetch(ticker, company, sector)

def \_fallback\_ratio\_fetch(ticker: str, company: str, sector: str) \-\> dict:

    """Direct yfinance fetch if ReAct agent fails."""

    try:

        from backend.tools.financial\_tools import get\_stock\_info as \_get, get\_sector\_median as \_med

        info \= \_get.func(ticker)

        meds \= \_med.func(sector)

        medians \= meds.get("medians", {})

        ratios \= \[\]

        fields \= \[

            ("P/E Ratio", info.get("pe\_ratio"), medians.get("pe"), "x"),

            ("P/B Ratio", info.get("pb\_ratio"), medians.get("pb"), "x"),

            ("ROE", info.get("roe\_pct"), medians.get("roe"), "%"),

            ("Operating Margin", info.get("operating\_margins\_pct"), medians.get("operating\_margin"), "%"),

            ("Net Profit Margin", info.get("profit\_margins\_pct"), medians.get("net\_margin"), "%"),

            ("Debt/Equity", info.get("debt\_to\_equity"), medians.get("de\_ratio"), "x"),

        \]

        for metric, value, median, unit in fields:

            if value is not None:

                if median and value and float(value) \> float(median) \* 1.1:

                    assessment \= f"Above sector median ({median}{unit})"

                elif median and value and float(value) \< float(median) \* 0.9:

                    assessment \= f"Below sector median ({median}{unit})"

                else:

                    assessment \= f"Inline with sector median ({median}{unit})" if median else "No sector benchmark available"

                ratios.append({

                    "metric": metric, "value": value,

                    "sector\_median": median, "unit": unit,

                    "assessment": assessment,

                })

        return {

            "status": "fallback\_success",

            "ratios": ratios,

            "current\_price": info.get("current\_price"),

            "market\_cap\_cr": info.get("market\_cap\_cr"),

            "company": company,

        }

    except Exception as e:

        return {"status": "error", "message": str(e), "company": company}

### 7.3 `backend/agents/news_sentinel.py`

"""

News Sentinel Agent.

Fetches Google News RSS, classifies sentiment per article,

clusters into themes, returns aggregate scores \+ event risk flags.

"""

import logging

import json

from langchain\_openai import ChatOpenAI

from backend.tools.news\_tools import fetch\_google\_news

logger \= logging.getLogger(\_\_name\_\_)

llm \= ChatOpenAI(model="gpt-4o-mini", temperature=0)

THEMES \= \["order\_wins", "quarterly\_results", "regulatory", "management\_change", "macro\_sector", "legal\_dispute", "other"\]

def run\_news\_sentinel(ticker: str, company: str) \-\> dict:

    """

    Fetch, classify, and cluster news for a company.

    Returns sentiment score, theme breakdown, top headlines, event risk flags.

    """

    logger.info(f"\[NewsSentinel\] Starting for {company}")

    articles \= fetch\_google\_news(company\_name=company, ticker=ticker, limit=20)

    if not articles:

        return {

            "status": "no\_news",

            "message": "No news articles found",

            "company": company,

            "overall\_sentiment": "NEUTRAL",

            "sentiment\_score": 0.0,

            "themes": {},

            "top\_headlines": \[\],

            "event\_risk\_flags": \[\],

        }

    \# Build classification prompt for all articles at once (batch to save API calls)

    articles\_text \= "\\n".join(\[

        f"{i+1}. TITLE: {a\['title'\]}\\n   SOURCE: {a\['source'\]}\\n   DATE: {a\['published'\]\[:20\]}\\n   SUMMARY: {a\['summary'\]\[:200\]}"

        for i, a in enumerate(articles\[:15\])

    \])

    prompt \= f"""You are a financial analyst classifying news articles about {company}.

Articles:

{articles\_text}

For EACH article, classify:

1\. sentiment: "POSITIVE", "NEGATIVE", or "NEUTRAL"

2\. score: float from \-1.0 (very negative) to \+1.0 (very positive)

3\. theme: one of {THEMES}

4\. is\_event\_risk: true if article mentions legal action, SEBI order, promoter selling, insider trading, major accident, or financial fraud

Return ONLY valid JSON array — one object per article, in order:

\[

  {{"id": 1, "sentiment": "POSITIVE", "score": 0.7, "theme": "order\_wins", "is\_event\_risk": false}},

  ...

\]"""

    try:

        response \= llm.invoke(prompt)

        content \= response.content.strip()

        if content.startswith("\`\`\`"):

            content \= content.split("\`\`\`")\[1\]

            if content.startswith("json"):

                content \= content\[4:\]

        classifications \= json.loads(content.strip())

        \# Aggregate results

        scored\_articles \= \[\]

        for i, clf in enumerate(classifications):

            if i \< len(articles):

                scored\_articles.append({

                    "title": articles\[i\]\["title"\],

                    "source": articles\[i\]\["source"\],

                    "published": articles\[i\]\["published"\]\[:20\],

                    "sentiment": clf.get("sentiment", "NEUTRAL"),

                    "score": clf.get("score", 0.0),

                    "theme": clf.get("theme", "other"),

                    "is\_event\_risk": clf.get("is\_event\_risk", False),

                })

        \# Overall sentiment score (average)

        scores \= \[a\["score"\] for a in scored\_articles\]

        overall\_score \= round(sum(scores) / len(scores), 3\) if scores else 0.0

        if overall\_score \> 0.3:

            sentiment\_label \= "POSITIVE"

        elif overall\_score \< \-0.3:

            sentiment\_label \= "NEGATIVE"

        else:

            sentiment\_label \= "NEUTRAL"

        \# Theme breakdown

        theme\_data \= {}

        for theme in THEMES:

            theme\_articles \= \[a for a in scored\_articles if a\["theme"\] \== theme\]

            if theme\_articles:

                theme\_score \= round(sum(a\["score"\] for a in theme\_articles) / len(theme\_articles), 3\)

                theme\_data\[theme\] \= {

                    "count": len(theme\_articles),

                    "avg\_score": theme\_score,

                    "sentiment": "POSITIVE" if theme\_score \> 0.2 else ("NEGATIVE" if theme\_score \< \-0.2 else "NEUTRAL"),

                    "sample\_headline": theme\_articles\[0\]\["title"\],

                }

        \# Event risk flags

        event\_risks \= \[a\["title"\] for a in scored\_articles if a.get("is\_event\_risk")\]

        \# Top headlines (sorted by abs score — most impactful first)

        top\_headlines \= sorted(scored\_articles, key=lambda x: abs(x\["score"\]), reverse=True)\[:5\]

        top\_headlines\_list \= \[f"{h\['title'\]} \[{h\['sentiment'\]}, {h\['source'\]}\]" for h in top\_headlines\]

        result \= {

            "status": "success",

            "company": company,

            "articles\_analyzed": len(scored\_articles),

            "overall\_sentiment": sentiment\_label,

            "sentiment\_score": overall\_score,

            "themes": theme\_data,

            "top\_headlines": top\_headlines\_list,

            "event\_risk\_flags": event\_risks,

            "news\_summary": (

                f"Analyzed {len(scored\_articles)} articles. Overall sentiment: {sentiment\_label} "

                f"(score: {overall\_score:+.2f}). "

                f"{'Event risk flags: ' \+ '; '.join(event\_risks\[:2\]) \+ '.' if event\_risks else 'No event risk flags identified.'}"

            ),

        }

        logger.info(f"\[NewsSentinel\] Complete for {company}: {sentiment\_label} ({overall\_score:+.2f})")

        return result

    except Exception as e:

        logger.error(f"\[NewsSentinel\] Error: {e}")

        return {

            "status": "error",

            "message": str(e),

            "company": company,

            "overall\_sentiment": "NEUTRAL",

            "sentiment\_score": 0.0,

            "themes": {},

            "top\_headlines": \[\],

            "event\_risk\_flags": \[\],

        }

### 7.4 `backend/agents/technical_analyst.py`

"""

Technical Analyst Agent.

Fetches 1-year OHLCV from yfinance, computes technical indicators via pandas\_ta,

generates natural language interpretation.

"""

import logging

from langchain\_openai import ChatOpenAI

from backend.tools.financial\_tools import get\_price\_history

from backend.tools.technical\_tools import compute\_technical\_indicators

logger \= logging.getLogger(\_\_name\_\_)

llm \= ChatOpenAI(model="gpt-4o-mini", temperature=0)

def run\_technical\_analyst(ticker: str, company: str) \-\> dict:

    """

    Compute technical indicators and generate interpretation.

    Returns structured dict with all indicators \+ summary.

    """

    logger.info(f"\[TechnicalAnalyst\] Starting for {company} ({ticker})")

    \# Fetch 1-year price history

    price\_data \= get\_price\_history.func(ticker, "1y")

    if "error" in price\_data:

        return {

            "status": "error",

            "message": price\_data\["error"\],

            "company": company,

        }

    \# Compute indicators

    indicators \= compute\_technical\_indicators(price\_data)

    if "error" in indicators:

        return {

            "status": "error",

            "message": indicators\["error"\],

            "company": company,

        }

    \# LLM interpretation

    prompt \= f"""You are a technical analyst reviewing {company}'s stock.

Computed indicators:

\- Current Price: ₹{indicators.get('current\_price')}

\- Trend: {indicators.get('trend')}

\- RSI (14-day): {indicators.get('rsi')} — {indicators.get('rsi\_signal')}

\- MACD: {indicators.get('macd')} vs Signal: {indicators.get('macd\_signal\_line')} — {indicators.get('macd\_direction')}

\- 20-day SMA: ₹{indicators.get('sma\_20')}

\- 50-day SMA: ₹{indicators.get('sma\_50')}

\- 200-day SMA: ₹{indicators.get('sma\_200')}

\- Golden Cross: {indicators.get('golden\_cross')} | Death Cross: {indicators.get('death\_cross')}

\- Support: ₹{indicators.get('support\_level')} | Resistance: ₹{indicators.get('resistance\_level')}

\- 52-week High: ₹{indicators.get('52w\_high')} | Low: ₹{indicators.get('52w\_low')}

\- 52-week Position: {indicators.get('52w\_position\_pct')}% (100% \= at 52w high)

\- 1-year price return: {price\_data.get('price\_return\_pct')}%

Write a 3–4 sentence technical analysis summary. Be specific. Include:

1\. The primary trend and its strength

2\. Momentum signals (RSI \+ MACD)

3\. Key price levels to watch

4\. An overall technical verdict (Bullish / Bearish / Neutral with conditions)"""

    try:

        response \= llm.invoke(prompt)

        indicators\["technical\_interpretation"\] \= response.content.strip()

        indicators\["status"\] \= "success"

        indicators\["company"\] \= company

        indicators\["ticker"\] \= ticker

        indicators\["price\_return\_1y\_pct"\] \= price\_data.get("price\_return\_pct")

        logger.info(f"\[TechnicalAnalyst\] Complete for {company}")

        return indicators

    except Exception as e:

        logger.error(f"\[TechnicalAnalyst\] LLM error: {e}")

        indicators\["technical\_interpretation"\] \= indicators.get("technicals\_summary", "Technical analysis computed.")

        indicators\["status"\] \= "partial\_success"

        indicators\["company"\] \= company

        return indicators

### 7.5 `backend/risk/metrics.py`

"""

Risk metrics computation for Risk Assessor Agent.

All computations are pure math — no LLM calls, no external APIs.

Reads from LangGraph state populated by previous agents.

"""

import logging

import numpy as np

import pandas as pd

logger \= logging.getLogger(\_\_name\_\_)

RISK\_FREE\_RATE \= 0.067   \# 10-year India G-Sec yield \~6.7%

PLEDGE\_THRESHOLD \= 25.0  \# Promoter pledge % above this \= red flag

BETA\_HIGH \= 1.5

BETA\_MEDIUM \= 1.2

VOL\_HIGH \= 0.35          \# 35% annualized

DRAWDOWN\_HIGH \= 0.40     \# 40% max drawdown

IC\_THRESHOLD \= 2.0       \# Interest coverage below this \= solvency risk

def compute\_market\_risk(close\_prices: list\[float\], nifty\_closes: list\[float\]) \-\> dict:

    """Compute Beta, Volatility, VaR, Max Drawdown, Sharpe from price series."""

    try:

        stock\_returns \= pd.Series(close\_prices).pct\_change().dropna()

        nifty\_returns \= pd.Series(nifty\_closes\[-len(close\_prices):\]).pct\_change().dropna()

        \# Align lengths

        min\_len \= min(len(stock\_returns), len(nifty\_returns))

        sr \= stock\_returns.iloc\[-min\_len:\]

        nr \= nifty\_returns.iloc\[-min\_len:\]

        \# Beta

        covariance \= np.cov(sr, nr)\[0\]\[1\]

        nifty\_variance \= np.var(nr)

        beta \= round(covariance / nifty\_variance, 3\) if nifty\_variance \> 0 else 1.0

        \# Annualized volatility

        ann\_vol \= round(float(sr.std() \* np.sqrt(252)), 4\)

        \# Historical VaR at 95% confidence (5th percentile)

        var\_95\_daily \= round(float(np.percentile(sr, 5)), 4\)

        var\_95\_pct \= f"{var\_95\_daily \* 100:.2f}%"

        \# Max Drawdown

        cumulative \= (1 \+ sr).cumprod()

        rolling\_max \= cumulative.cummax()

        drawdown \= (rolling\_max \- cumulative) / rolling\_max

        max\_drawdown \= round(float(drawdown.max()), 4\)

        \# Sharpe Ratio

        mean\_daily\_return \= sr.mean()

        sharpe \= round(

            (mean\_daily\_return \* 252 \- RISK\_FREE\_RATE) / ann\_vol, 3

        ) if ann\_vol \> 0 else 0.0

        return {

            "beta": beta,

            "annualized\_volatility\_pct": round(ann\_vol \* 100, 2),

            "var\_95\_daily\_pct": var\_95\_pct,

            "max\_drawdown\_pct": round(max\_drawdown \* 100, 2),

            "sharpe\_ratio": sharpe,

        }

    except Exception as e:

        logger.error(f"Market risk computation error: {e}")

        return {}

def compute\_altman\_z(ratio\_output: dict, info: dict) \-\> dict:

    """

    Compute Altman Z-Score from available financial data.

    Z \> 2.99: Safe | 1.81-2.99: Grey zone | \< 1.81: Distress zone

    Uses simplified form with available yfinance data.

    """

    try:

        price \= info.get("current\_price", 0\) or 0

        shares \= info.get("shares\_outstanding", 0\) or 0

        market\_cap \= price \* shares

        total\_assets \= info.get("total\_assets")  \# Not always available from yfinance

        total\_debt \= info.get("total\_debt") or 0

        total\_cash \= info.get("total\_cash") or 0

        net\_income \= info.get("net\_income") or 0

        revenue \= info.get("revenue") or 0

        de\_ratio \= info.get("debt\_to\_equity") or 0

        \# If total\_assets not available, estimate from market\_cap \+ total\_debt

        if not total\_assets and market\_cap \> 0:

            total\_assets \= market\_cap \+ total\_debt \- total\_cash

        if not total\_assets or total\_assets \<= 0:

            return {"altman\_z": None, "altman\_zone": "UNKNOWN", "note": "Insufficient data for Altman Z"}

        working\_capital \= total\_cash \* 0.7  \# Rough proxy

        retained\_earnings \= net\_income \* 3   \# Rough proxy (3-year accumulation)

        ebit \= net\_income \* 1.3              \# Rough proxy (add \~30% for taxes)

        total\_liabilities \= total\_debt

        X1 \= working\_capital / total\_assets

        X2 \= retained\_earnings / total\_assets

        X3 \= ebit / total\_assets

        X4 \= market\_cap / total\_liabilities if total\_liabilities \> 0 else 10.0

        X5 \= revenue / total\_assets

        z\_score \= round(1.2\*X1 \+ 1.4\*X2 \+ 3.3\*X3 \+ 0.6\*X4 \+ 1.0\*X5, 3\)

        if z\_score \> 2.99:

            zone \= "SAFE"

        elif z\_score \> 1.81:

            zone \= "GREY"

        else:

            zone \= "DISTRESS"

        return {

            "altman\_z": z\_score,

            "altman\_zone": zone,

            "note": "Approximate score using available data. Full computation requires audited balance sheet."

        }

    except Exception as e:

        logger.error(f"Altman Z computation error: {e}")

        return {"altman\_z": None, "altman\_zone": "UNKNOWN"}

def build\_risk\_scorecard(

    market\_risk: dict,

    ratio\_output: dict,

    filing\_output: dict,

    news\_output: dict,

    screener\_data: dict,

    altman: dict,

) \-\> dict:

    """Build the full risk scorecard and determine overall risk level."""

    red\_flags \= \[\]

    amber\_flags \= \[\]

    \# \--- Market risk flags \---

    beta \= market\_risk.get("beta", 1.0)

    vol \= market\_risk.get("annualized\_volatility\_pct", 0\)

    drawdown \= market\_risk.get("max\_drawdown\_pct", 0\)

    sharpe \= market\_risk.get("sharpe\_ratio", 0\)

    if beta and beta \> BETA\_HIGH:

        red\_flags.append(f"High Beta ({beta}x) — significantly more volatile than market")

    elif beta and beta \> BETA\_MEDIUM:

        amber\_flags.append(f"Elevated Beta ({beta}x) — moderately above market volatility")

    if vol \> VOL\_HIGH \* 100:

        amber\_flags.append(f"High annualized volatility ({vol}%) — elevated price risk")

    if drawdown \> DRAWDOWN\_HIGH \* 100:

        red\_flags.append(f"Large max drawdown ({drawdown}%) in past year")

    if sharpe \< 0.5:

        amber\_flags.append(f"Low Sharpe ratio ({sharpe}) — poor risk-adjusted return")

    \# \--- Solvency risk flags \---

    de\_ratio \= None

    for r in ratio\_output.get("ratios", \[\]):

        if r.get("metric") \== "Debt/Equity":

            de\_ratio \= r.get("value")

            break

    if de\_ratio and float(de\_ratio) \> 2.0:

        red\_flags.append(f"High Debt/Equity ({de\_ratio}x) — elevated leverage risk")

    elif de\_ratio and float(de\_ratio) \> 1.5:

        amber\_flags.append(f"Moderate Debt/Equity ({de\_ratio}x)")

    \# Altman Z-Score

    if altman.get("altman\_z"):

        zone \= altman.get("altman\_zone")

        z \= altman.get("altman\_z")

        if zone \== "DISTRESS":

            red\_flags.append(f"Altman Z-Score {z} — in financial distress zone (\<1.81)")

        elif zone \== "GREY":

            amber\_flags.append(f"Altman Z-Score {z} — in grey zone (1.81–2.99)")

    \# \--- India-specific flags \---

    pledge\_pct \= screener\_data.get("pledge\_pct", 0\) or 0

    promoter\_pct \= screener\_data.get("promoter\_holding", 0\) or 0

    if float(pledge\_pct) \> PLEDGE\_THRESHOLD:

        red\_flags.append(

            f"HIGH PROMOTER PLEDGE ({pledge\_pct}%) — forced selling risk if price falls"

        )

    elif float(pledge\_pct) \> 10:

        amber\_flags.append(f"Moderate promoter pledge ({pledge\_pct}%)")

    if float(promoter\_pct) \< 35:

        amber\_flags.append(f"Low promoter holding ({promoter\_pct}%) — limited insider confidence")

    \# \--- News risk flags \---

    event\_risks \= news\_output.get("event\_risk\_flags", \[\])

    for risk in event\_risks\[:2\]:

        amber\_flags.append(f"News event risk: {risk\[:80\]}")

    \# \--- Earnings quality \---

    \# (Would ideally use CFO/NetProfit but requires full cash flow statement)

    stated\_risks \= filing\_output.get("stated\_risks", \[\])

    if len(stated\_risks) \> 5:

        amber\_flags.append(f"{len(stated\_risks)} risk factors explicitly mentioned in filings")

    \# \--- Overall risk level \---

    red\_count \= len(red\_flags)

    amber\_count \= len(amber\_flags)

    if red\_count \>= 3:

        risk\_level \= "VERY HIGH"

    elif red\_count \>= 2:

        risk\_level \= "HIGH"

    elif red\_count \>= 1 or amber\_count \>= 4:

        risk\_level \= "MEDIUM-HIGH"

    elif amber\_count \>= 2:

        risk\_level \= "MEDIUM"

    else:

        risk\_level \= "LOW"

    return {

        "risk\_level": risk\_level,

        "red\_flags": red\_flags,

        "amber\_flags": amber\_flags,

        "market\_risk": market\_risk,

        "altman\_z\_score": altman.get("altman\_z"),

        "altman\_zone": altman.get("altman\_zone"),

        "promoter\_pledge\_pct": pledge\_pct,

        "promoter\_holding\_pct": promoter\_pct,

        "risk\_summary": (

            f"Risk Level: {risk\_level}. "

            f"{len(red\_flags)} red flag(s), {len(amber\_flags)} amber flag(s). "

            f"{'Key concern: ' \+ red\_flags\[0\] if red\_flags else 'No critical red flags identified.'}"

        ),

    }

### 7.6 `backend/agents/risk_assessor.py`

"""

Risk Assessor Agent.

Reads all previous agent outputs from LangGraph state.

Computes comprehensive risk scorecard — no external API calls.

"""

import logging

import yfinance as yf

from backend.risk.metrics import compute\_market\_risk, compute\_altman\_z, build\_risk\_scorecard

from backend.tools.financial\_tools import get\_nifty\_history

logger \= logging.getLogger(\_\_name\_\_)

def run\_risk\_assessor(

    ticker: str,

    company: str,

    filing\_output: dict,

    ratio\_output: dict,

    news\_output: dict,

    technical\_output: dict,

) \-\> dict:

    """

    Compute the full risk scorecard from all agent outputs.

    Returns structured risk dict with overall level, flags, and metrics.

    """

    logger.info(f"\[RiskAssessor\] Starting for {company}")

    \# Get price history for market risk metrics

    close\_prices \= technical\_output.get("close") if isinstance(technical\_output.get("close"), list) else \[\]

    \# If technical\_output doesn't have price list, fetch directly

    if not close\_prices:

        try:

            stock \= yf.Ticker(ticker if ticker.endswith(".NS") else ticker \+ ".NS")

            hist \= stock.history(period="1y")

            close\_prices \= hist\["Close"\].tolist()

        except Exception as e:

            logger.warning(f"Price fetch for risk: {e}")

            close\_prices \= \[\]

    \# Get Nifty for Beta

    nifty\_data \= get\_nifty\_history.func("1y")

    nifty\_closes \= nifty\_data.get("close", \[\])

    \# Compute market risk metrics

    market\_risk \= {}

    if close\_prices and nifty\_closes:

        market\_risk \= compute\_market\_risk(close\_prices, nifty\_closes)

    \# Get screener data for India-specific flags

    screener\_data \= {}

    try:

        import pandas as pd

        from pathlib import Path

        screener\_file \= Path(f"data/screener\_cache/{company.upper()}.xlsx")

        if screener\_file.exists():

            df \= pd.read\_excel(screener\_file)

            \# Try to find pledge and promoter columns

            cols \= {str(c).lower(): c for c in df.columns}

            screener\_data \= {

                "promoter\_holding": \_safe\_first(df, "promoter holding"),

                "pledge\_pct": \_safe\_first(df, "pledge"),

                "roce": \_safe\_first(df, "roce"),

                "sales\_growth": \_safe\_first(df, "sales growth"),

            }

    except Exception as e:

        logger.warning(f"Screener data read error: {e}")

    \# Compute Altman Z

    try:

        stock\_info \= yf.Ticker(ticker if ticker.endswith(".NS") else ticker \+ ".NS").info

    except Exception:

        stock\_info \= {}

    altman \= compute\_altman\_z(ratio\_output, {

        "current\_price": technical\_output.get("current\_price"),

        "shares\_outstanding": stock\_info.get("sharesOutstanding"),

        "total\_debt": stock\_info.get("totalDebt"),

        "total\_cash": stock\_info.get("totalCash"),

        "net\_income": stock\_info.get("netIncomeToCommon"),

        "revenue": stock\_info.get("totalRevenue"),

        "debt\_to\_equity": stock\_info.get("debtToEquity"),

    })

    \# Build full scorecard

    scorecard \= build\_risk\_scorecard(

        market\_risk=market\_risk,

        ratio\_output=ratio\_output,

        filing\_output=filing\_output,

        news\_output=news\_output,

        screener\_data=screener\_data,

        altman=altman,

    )

    scorecard\["status"\] \= "success"

    scorecard\["company"\] \= company

    logger.info(f"\[RiskAssessor\] Complete for {company}: {scorecard\['risk\_level'\]}")

    return scorecard

def \_safe\_first(df, col\_fragment: str):

    """Find a column containing col\_fragment and return its first numeric value."""

    try:

        matching \= \[c for c in df.columns if col\_fragment.lower() in str(c).lower()\]

        if matching:

            val \= df\[matching\[0\]\].dropna().iloc\[-1\]

            return float(val)

    except Exception:

        pass

    return None

### 7.7 `backend/agents/report_writer.py`

"""

Report Writer Agent.

Uses GPT-4o for final synthesis.

Mandatory critique step before writing the report.

Produces structured Bull/Bear/Risk/Verdict markdown report.

"""

import logging

import json

from langchain\_openai import ChatOpenAI

logger \= logging.getLogger(\_\_name\_\_)

\# Use GPT-4o for best quality synthesis

llm \= ChatOpenAI(model="gpt-4o", temperature=0.2)

REPORT\_TEMPLATE \= """

\# {company} — Equity Research Brief

\*Generated by FinSight AI | Academic prototype only\*

\#\# Company Snapshot

{snapshot}

\---

\#\# Inter-Agent Critique Resolution

\*\*Identified conflict:\*\* {conflict}

\*\*Resolution:\*\* {resolution}

\---

\#\# Bull Case 🟢

{bull\_case}

\#\# Bear Case 🔴

{bear\_case}

\---

\#\# Key Financial Metrics

{metrics\_table}

\---

\#\# Risk Assessment

\*\*Risk Level: {risk\_level}\*\*

{risk\_flags}

\---

\#\# Analyst Verdict

\*\*Rating:\*\* {verdict}

\*\*Confidence:\*\* {confidence}

{verdict\_rationale}

\---

\*Disclaimer: This report is generated by an AI research prototype (FinSight AI) for academic demonstration purposes only. Not SEBI-registered. Not financial advice. Data accuracy not guaranteed. Do not make investment decisions based on this output.\*

"""

def run\_report\_writer(

    ticker: str,

    company: str,

    sector: str,

    filing\_output: dict,

    ratio\_output: dict,

    news\_output: dict,

    technical\_output: dict,

    risk\_output: dict,

) \-\> dict:

    """

    Synthesize all agent outputs into a final investment research report.

    Runs critique step first, then generates full report.

    """

    logger.info(f"\[ReportWriter\] Starting final synthesis for {company}")

    \# ── STEP 1: CRITIQUE ──────────────────────────────────────────────────

    critique\_prompt \= f"""You are reviewing a multi-agent equity research system's outputs for {company} ({ticker}).

FILING ANALYSIS: {json.dumps(filing\_output, indent=2)\[:1500\]}

FINANCIAL RATIOS: {json.dumps(ratio\_output, indent=2)\[:1000\]}

NEWS SENTIMENT: Overall: {news\_output.get('overall\_sentiment')} (score: {news\_output.get('sentiment\_score')})

Top headlines: {news\_output.get('top\_headlines', \[\])\[:3\]}

Event risks: {news\_output.get('event\_risk\_flags', \[\])}

TECHNICAL: {technical\_output.get('trend')} | RSI: {technical\_output.get('rsi')} | MACD: {technical\_output.get('macd\_direction')}

RISK SCORECARD: Level: {risk\_output.get('risk\_level')} | Red flags: {risk\_output.get('red\_flags', \[\])}

Identify the SINGLE most significant conflict or tension between any two agents' conclusions.

Return JSON:

{{

  "conflict\_identified": "one sentence describing the conflict (e.g. 'Filing Analyst shows strong revenue growth while Risk Agent flags high debt leverage')",

  "agent\_a": "agent name",

  "agent\_b": "agent name",

  "resolution": "one sentence on how to reconcile this in the final report"

}}"""

    try:

        critique\_response \= llm.invoke(critique\_prompt)

        critique\_text \= critique\_response.content.strip()

        if "\`\`\`" in critique\_text:

            critique\_text \= critique\_text.split("\`\`\`")\[1\]

            if critique\_text.startswith("json"):

                critique\_text \= critique\_text\[4:\]

        critique \= json.loads(critique\_text.strip())

    except Exception as e:

        logger.warning(f"\[ReportWriter\] Critique parse error: {e}")

        critique \= {

            "conflict\_identified": "No significant conflict identified across agent outputs.",

            "resolution": "All agents' conclusions are broadly consistent.",

        }

    \# ── STEP 2: FULL REPORT ───────────────────────────────────────────────

    report\_prompt \= f"""Write a professional equity research report for {company} ({ticker}) listed on NSE.

Use the following verified data from specialist agents:

\*\*FILING INSIGHTS (from company's actual annual reports via RAG):\*\*

\- Revenue trend: {filing\_output.get('revenue\_trend', 'N/A')}

\- Margin: {filing\_output.get('ebitda\_margin', 'N/A')}

\- Management themes: {filing\_output.get('management\_themes', \[\])}

\- Stated risks: {filing\_output.get('stated\_risks', \[\])}

\- Overall: {filing\_output.get('overall\_filing\_assessment', 'N/A')}

\*\*FINANCIAL RATIOS (computed from live yfinance data):\*\*

{chr(10).join(\[f"- {r\['metric'\]}: {r\['value'\]} (Sector median: {r.get('sector\_median', 'N/A')}) — {r\['assessment'\]}" for r in ratio\_output.get('ratios', \[\])\[:6\]\])}

Valuation verdict: {ratio\_output.get('valuation\_verdict', 'N/A')}

\*\*NEWS INTELLIGENCE (from Google News RSS):\*\*

\- Overall sentiment: {news\_output.get('overall\_sentiment')} (score: {news\_output.get('sentiment\_score')})

\- Top stories: {'; '.join(news\_output.get('top\_headlines', \[\])\[:3\])}

\- Event risk flags: {news\_output.get('event\_risk\_flags', 'None')}

\*\*TECHNICAL ANALYSIS (from 1-year OHLCV data):\*\*

{technical\_output.get('technical\_interpretation', technical\_output.get('technicals\_summary', 'N/A'))}

Key levels: Support ₹{technical\_output.get('support\_level')}, Resistance ₹{technical\_output.get('resistance\_level')}

\*\*RISK SCORECARD:\*\*

Risk Level: {risk\_output.get('risk\_level')}

Red flags: {risk\_output.get('red\_flags', \[\])}

Beta: {risk\_output.get('market\_risk', {}).get('beta')} | Volatility: {risk\_output.get('market\_risk', {}).get('annualized\_volatility\_pct')}% | Altman Z: {risk\_output.get('altman\_z\_score')} ({risk\_output.get('altman\_zone')})

Promoter pledge: {risk\_output.get('promoter\_pledge\_pct')}%

\*\*CRITIQUE RESOLUTION:\*\*

Conflict: {critique.get('conflict\_identified')}

Resolution: {critique.get('resolution')}

Write this report in structured markdown:

\#\#\# Company Snapshot

\[2–3 sentences: what the company does, its size, why it matters\]

\#\#\# Bull Case

• \*\*\[specific point\]\*\*: \[evidence from agents — cite which data\]

• \*\*\[specific point\]\*\*: \[evidence\]

• \*\*\[specific point\]\*\*: \[evidence\]

\#\#\# Bear Case

• \*\*\[specific point\]\*\*: \[evidence from agents\]

• \*\*\[specific point\]\*\*: \[evidence\]

• \*\*\[specific point\]\*\*: \[evidence\]

\#\#\# Metrics Summary

| Metric | Value | Benchmark | Assessment |

|--------|-------|-----------|------------|

\[Use actual ratio data\]

\#\#\# Risk Flags

\[List actual flags from risk scorecard\]

\#\#\# Verdict

Rating: \[BUY / ACCUMULATE / HOLD / REDUCE / SELL\]

Confidence: \[LOW / MEDIUM / HIGH\]

\[2 sentences of rationale referencing specific evidence\]

Be specific and evidence-linked. Do not add generic statements."""

    try:

        report\_response \= llm.invoke(report\_prompt)

        report\_content \= report\_response.content.strip()

        logger.info(f"\[ReportWriter\] Report generated for {company} ({len(report\_content)} chars)")

        return {

            "status": "success",

            "company": company,

            "ticker": ticker,

            "report\_markdown": report\_content,

            "critique": critique,

            "risk\_level": risk\_output.get("risk\_level"),

            "overall\_sentiment": news\_output.get("overall\_sentiment"),

        }

    except Exception as e:

        logger.error(f"\[ReportWriter\] Error: {e}")

        return {

            "status": "error",

            "message": str(e),

            "company": company,

        }

---

## 8\. Phase 4 — LangGraph Orchestration

### 8.1 `backend/graph/state.py`

"""

FinSightState — the shared state object for the LangGraph pipeline.

Every agent reads from and writes to this TypedDict.

"""

from typing import TypedDict, Optional, Any

class FinSightState(TypedDict):

    \# ── INPUT ──────────────────────────────────────────────

    ticker: str              \# e.g. "RELIANCE.NS"

    company\_name: str        \# e.g. "Reliance Industries"

    sector: str              \# e.g. "Energy"

    \# ── AGENT OUTPUTS (None until that agent completes) ────

    filing\_output: Optional\[dict\]

    ratio\_output: Optional\[dict\]

    news\_output: Optional\[dict\]

    technical\_output: Optional\[dict\]

    risk\_output: Optional\[dict\]

    \# ── FINAL OUTPUT ───────────────────────────────────────

    final\_report: Optional\[dict\]   \# Full report dict from Report Writer

    \# ── METADATA ──────────────────────────────────────────

    errors: list\[str\]        \# Non-fatal errors (pipeline continues)

    warnings: list\[str\]      \# Warnings logged but not blocking

### 8.2 `backend/graph/workflow.py`

"""

LangGraph StateGraph for FinSight AI.

Parallel fan-out to 4 specialist agents, then Risk Assessor, then Report Writer.

"""

import logging

from langgraph.graph import StateGraph, START, END

from langgraph.constants import Send

from backend.graph.state import FinSightState

from backend.agents.filing\_analyst import run\_filing\_analyst

from backend.agents.ratio\_cruncher import run\_ratio\_cruncher

from backend.agents.news\_sentinel import run\_news\_sentinel

from backend.agents.technical\_analyst import run\_technical\_analyst

from backend.agents.risk\_assessor import run\_risk\_assessor

from backend.agents.report\_writer import run\_report\_writer

from backend.utils.sector\_data import detect\_sector\_from\_yfinance

logger \= logging.getLogger(\_\_name\_\_)

\# ── NODE FUNCTIONS ────────────────────────────────────────────────────────

\# Each node takes state, does its work, returns dict with ONLY the fields it updates

def filing\_analyst\_node(state: FinSightState) \-\> dict:

    logger.info("=== FILING ANALYST NODE \===")

    try:

        result \= run\_filing\_analyst(state\["ticker"\], state\["company\_name"\])

        return {"filing\_output": result}

    except Exception as e:

        logger.error(f"Filing analyst node error: {e}")

        return {

            "filing\_output": {"status": "error", "message": str(e)},

            "errors": state.get("errors", \[\]) \+ \[f"FilingAnalyst: {str(e)}"\],

        }

def ratio\_cruncher\_node(state: FinSightState) \-\> dict:

    logger.info("=== RATIO CRUNCHER NODE \===")

    try:

        result \= run\_ratio\_cruncher(state\["ticker"\], state\["company\_name"\], state\["sector"\])

        return {"ratio\_output": result}

    except Exception as e:

        logger.error(f"Ratio cruncher node error: {e}")

        return {

            "ratio\_output": {"status": "error", "message": str(e), "ratios": \[\]},

            "errors": state.get("errors", \[\]) \+ \[f"RatioCruncher: {str(e)}"\],

        }

def news\_sentinel\_node(state: FinSightState) \-\> dict:

    logger.info("=== NEWS SENTINEL NODE \===")

    try:

        result \= run\_news\_sentinel(state\["ticker"\], state\["company\_name"\])

        return {"news\_output": result}

    except Exception as e:

        logger.error(f"News sentinel node error: {e}")

        return {

            "news\_output": {"status": "error", "overall\_sentiment": "NEUTRAL", "sentiment\_score": 0.0},

            "errors": state.get("errors", \[\]) \+ \[f"NewsSentinel: {str(e)}"\],

        }

def technical\_analyst\_node(state: FinSightState) \-\> dict:

    logger.info("=== TECHNICAL ANALYST NODE \===")

    try:

        result \= run\_technical\_analyst(state\["ticker"\], state\["company\_name"\])

        return {"technical\_output": result}

    except Exception as e:

        logger.error(f"Technical analyst node error: {e}")

        return {

            "technical\_output": {"status": "error", "message": str(e)},

            "errors": state.get("errors", \[\]) \+ \[f"TechnicalAnalyst: {str(e)}"\],

        }

def risk\_assessor\_node(state: FinSightState) \-\> dict:

    logger.info("=== RISK ASSESSOR NODE \===")

    try:

        result \= run\_risk\_assessor(

            ticker=state\["ticker"\],

            company=state\["company\_name"\],

            filing\_output=state.get("filing\_output") or {},

            ratio\_output=state.get("ratio\_output") or {},

            news\_output=state.get("news\_output") or {},

            technical\_output=state.get("technical\_output") or {},

        )

        return {"risk\_output": result}

    except Exception as e:

        logger.error(f"Risk assessor node error: {e}")

        return {

            "risk\_output": {"status": "error", "risk\_level": "UNKNOWN", "red\_flags": \[\], "amber\_flags": \[\]},

            "errors": state.get("errors", \[\]) \+ \[f"RiskAssessor: {str(e)}"\],

        }

def report\_writer\_node(state: FinSightState) \-\> dict:

    logger.info("=== REPORT WRITER NODE \===")

    try:

        result \= run\_report\_writer(

            ticker=state\["ticker"\],

            company=state\["company\_name"\],

            sector=state\["sector"\],

            filing\_output=state.get("filing\_output") or {},

            ratio\_output=state.get("ratio\_output") or {"ratios": \[\]},

            news\_output=state.get("news\_output") or {"overall\_sentiment": "NEUTRAL", "sentiment\_score": 0.0},

            technical\_output=state.get("technical\_output") or {},

            risk\_output=state.get("risk\_output") or {"risk\_level": "UNKNOWN"},

        )

        return {"final\_report": result}

    except Exception as e:

        logger.error(f"Report writer node error: {e}")

        return {

            "final\_report": {"status": "error", "message": str(e)},

            "errors": state.get("errors", \[\]) \+ \[f"ReportWriter: {str(e)}"\],

        }

\# ── ROUTING FUNCTION ──────────────────────────────────────────────────────

def route\_to\_parallel\_agents(state: FinSightState):

    """

    Fan-out to all 4 specialist agents simultaneously.

    LangGraph runs all Send() targets in parallel.

    """

    return \[

        Send("filing\_analyst", state),

        Send("ratio\_cruncher", state),

        Send("news\_sentinel", state),

        Send("technical\_analyst", state),

    \]

\# ── BUILD GRAPH ───────────────────────────────────────────────────────────

def build\_finsight\_graph():

    builder \= StateGraph(FinSightState)

    \# Add all nodes

    builder.add\_node("filing\_analyst",   filing\_analyst\_node)

    builder.add\_node("ratio\_cruncher",   ratio\_cruncher\_node)

    builder.add\_node("news\_sentinel",    news\_sentinel\_node)

    builder.add\_node("technical\_analyst", technical\_analyst\_node)

    builder.add\_node("risk\_assessor",    risk\_assessor\_node)

    builder.add\_node("report\_writer",    report\_writer\_node)

    \# START → parallel fan-out to all 4 specialist agents

    builder.add\_conditional\_edges(START, route\_to\_parallel\_agents)

    \# All 4 specialist agents → risk assessor (LangGraph waits for all 4\)

    builder.add\_edge("filing\_analyst",    "risk\_assessor")

    builder.add\_edge("ratio\_cruncher",    "risk\_assessor")

    builder.add\_edge("news\_sentinel",     "risk\_assessor")

    builder.add\_edge("technical\_analyst", "risk\_assessor")

    \# Risk assessor → report writer → END

    builder.add\_edge("risk\_assessor",  "report\_writer")

    builder.add\_edge("report\_writer",  END)

    return builder.compile()

\# Singleton graph instance

finsight\_graph \= build\_finsight\_graph()

def run\_analysis(ticker: str, company\_name: str, sector: str \= None) \-\> FinSightState:

    """

    Run the full FinSight AI pipeline.

    Returns the final state with all agent outputs.

    """

    \# Auto-detect sector if not provided

    if not sector:

        try:

            import yfinance as yf

            ns\_ticker \= ticker if ticker.endswith(".NS") else ticker \+ ".NS"

            info \= yf.Ticker(ns\_ticker).info

            yf\_sector \= info.get("sector", "")

            sector \= detect\_sector\_from\_yfinance(yf\_sector)

        except Exception:

            sector \= "Diversified"

    initial\_state: FinSightState \= {

        "ticker": ticker if ticker.endswith(".NS") else ticker \+ ".NS",

        "company\_name": company\_name,

        "sector": sector,

        "filing\_output": None,

        "ratio\_output": None,

        "news\_output": None,

        "technical\_output": None,

        "risk\_output": None,

        "final\_report": None,

        "errors": \[\],

        "warnings": \[\],

    }

    logger.info(f"Starting FinSight AI pipeline for {company\_name} ({ticker})")

    result \= finsight\_graph.invoke(initial\_state)

    logger.info(f"Pipeline complete for {company\_name}. Errors: {result.get('errors', \[\])}")

    return result

---

## 9\. Phase 5 — FastAPI Backend

### 9.1 `main.py`

"""

FinSight AI — FastAPI Application Entry Point

Run: uvicorn main:app \--host 0.0.0.0 \--port 8000 \--reload

"""

import logging

import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, field\_validator

from dotenv import load\_dotenv

load\_dotenv()

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",

)

logger \= logging.getLogger(\_\_name\_\_)

\# Company registry — known companies with their NSE tickers and sectors

COMPANY\_REGISTRY \= {

    "RELIANCE":    {"name": "Reliance Industries",   "sector": "Energy"},

    "HDFCBANK":    {"name": "HDFC Bank",             "sector": "BFSI"},

    "INFY":        {"name": "Infosys",               "sector": "IT"},

    "TCS":         {"name": "TCS",                   "sector": "IT"},

    "WIPRO":       {"name": "Wipro",                 "sector": "IT"},

    "BEL":         {"name": "Bharat Electronics",    "sector": "Defence"},

    "TATAPOWER":   {"name": "Tata Power",            "sector": "Energy"},

    "BAJFINANCE":  {"name": "Bajaj Finance",         "sector": "NBFC"},

    "ADANIPORTS":  {"name": "Adani Ports",           "sector": "Infra"},

    "HINDUNILVR":  {"name": "Hindustan Unilever",    "sector": "FMCG"},

    "ICICIBANK":   {"name": "ICICI Bank",            "sector": "BFSI"},

    "SBIN":        {"name": "State Bank of India",   "sector": "BFSI"},

    "MARUTI":      {"name": "Maruti Suzuki",         "sector": "Auto"},

    "TATAMOTORS":  {"name": "Tata Motors",           "sector": "Auto"},

    "SUNPHARMA":   {"name": "Sun Pharmaceutical",    "sector": "Pharma"},

}

@asynccontextmanager

async def lifespan(app: FastAPI):

    logger.info("FinSight AI starting up...")

    \# Pre-index any companies that have PDFs downloaded

    try:

        from backend.rag.indexer import ingest\_all\_companies

        results \= ingest\_all\_companies(force\_reindex=False)

        indexed \= sum(1 for v in results.values() if v)

        logger.info(f"RAG indexing: {indexed}/{len(results)} companies indexed")

    except Exception as e:

        logger.warning(f"RAG pre-indexing error (non-fatal): {e}")

    yield

    logger.info("FinSight AI shutting down...")

app \= FastAPI(

    title="FinSight AI",

    description="Multi-Agent Equity Research Platform for Indian Stocks",

    version="1.0.0",

    lifespan=lifespan,

)

app.add\_middleware(

    CORSMiddleware,

    allow\_origins=\["\*"\],

    allow\_methods=\["\*"\],

    allow\_headers=\["\*"\],

)

\# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────

class AnalysisRequest(BaseModel):

    ticker: str

    company\_name: str \= ""

    sector: str \= ""

    @field\_validator("ticker")

    @classmethod

    def normalize\_ticker(cls, v: str) \-\> str:

        v \= v.strip().upper()

        if not v.endswith(".NS") and not v.endswith(".BO"):

            v \= v \+ ".NS"

        return v

    def resolve\_company\_info(self):

        """Fill company\_name and sector from registry if not provided."""

        base \= self.ticker.replace(".NS", "").replace(".BO", "")

        if base in COMPANY\_REGISTRY and not self.company\_name:

            self.company\_name \= COMPANY\_REGISTRY\[base\]\["name"\]

        if base in COMPANY\_REGISTRY and not self.sector:

            self.sector \= COMPANY\_REGISTRY\[base\]\["sector"\]

        if not self.company\_name:

            self.company\_name \= base  \# Fallback: use ticker as name

class AnalysisResponse(BaseModel):

    ticker: str

    company\_name: str

    sector: str

    status: str

    report\_markdown: str \= ""

    filing\_summary: dict \= {}

    ratios: list \= \[\]

    news\_sentiment: dict \= {}

    technical: dict \= {}

    risk\_scorecard: dict \= {}

    errors: list \= \[\]

    processing\_time\_seconds: float \= 0.0

\# ── ENDPOINTS ─────────────────────────────────────────────────────────────

@app.get("/")

def root():

    return {

        "name": "FinSight AI",

        "version": "1.0.0",

        "description": "Multi-Agent Equity Research Platform",

        "endpoints": {

            "POST /analyze": "Run full analysis pipeline",

            "GET /companies": "List supported companies",

            "GET /health": "Health check",

        },

    }

@app.get("/health")

def health():

    return {"status": "running", "service": "FinSight AI"}

@app.get("/companies")

def list\_companies():

    return {"companies": COMPANY\_REGISTRY}

@app.post("/analyze", response\_model=AnalysisResponse)

def analyze\_stock(request: AnalysisRequest):

    """

    Run the full 6-agent FinSight AI pipeline for a given stock ticker.

    """

    request.resolve\_company\_info()

    if not request.company\_name:

        raise HTTPException(status\_code=400, detail="company\_name is required if ticker is not in registry")

    logger.info(f"API: analyze request for {request.ticker} ({request.company\_name})")

    start\_time \= time.time()

    try:

        from backend.graph.workflow import run\_analysis

        state \= run\_analysis(

            ticker=request.ticker,

            company\_name=request.company\_name,

            sector=request.sector or "Diversified",

        )

    except Exception as e:

        logger.error(f"Pipeline error: {e}")

        raise HTTPException(status\_code=500, detail=f"Analysis pipeline error: {str(e)}")

    elapsed \= round(time.time() \- start\_time, 2\)

    final\_report \= state.get("final\_report") or {}

    return AnalysisResponse(

        ticker=request.ticker,

        company\_name=request.company\_name,

        sector=request.sector or "Diversified",

        status=final\_report.get("status", "unknown"),

        report\_markdown=final\_report.get("report\_markdown", ""),

        filing\_summary={

            k: v for k, v in (state.get("filing\_output") or {}).items()

            if k in \["revenue\_trend", "ebitda\_margin", "management\_themes",

                     "stated\_risks", "overall\_filing\_assessment"\]

        },

        ratios=(state.get("ratio\_output") or {}).get("ratios", \[\]),

        news\_sentiment={

            k: v for k, v in (state.get("news\_output") or {}).items()

            if k in \["overall\_sentiment", "sentiment\_score", "themes",

                     "top\_headlines", "event\_risk\_flags"\]

        },

        technical={

            k: v for k, v in (state.get("technical\_output") or {}).items()

            if k in \["trend", "rsi", "rsi\_signal", "macd\_direction",

                     "sma\_20", "sma\_50", "support\_level", "resistance\_level",

                     "52w\_position\_pct", "technical\_interpretation"\]

        },

        risk\_scorecard={

            k: v for k, v in (state.get("risk\_output") or {}).items()

            if k in \["risk\_level", "red\_flags", "amber\_flags", "market\_risk",

                     "altman\_z\_score", "altman\_zone", "promoter\_pledge\_pct",

                     "promoter\_holding\_pct", "risk\_summary"\]

        },

        errors=state.get("errors", \[\]),

        processing\_time\_seconds=elapsed,

    )

---

## 10\. Phase 6 — Gradio Frontend

### 10.1 `frontend/app.py`

"""

FinSight AI — Gradio Web Interface

Run: python frontend/app.py

"""

import gradio as gr

import requests

import json

API\_URL \= "http://localhost:8000"

EXAMPLE\_COMPANIES \= {

    "RELIANCE": "Reliance Industries",

    "HDFCBANK": "HDFC Bank",

    "INFY": "Infosys",

    "TCS": "TCS",

    "WIPRO": "Wipro",

    "BEL": "Bharat Electronics",

    "TATAPOWER": "Tata Power",

    "BAJFINANCE": "Bajaj Finance",

    "ADANIPORTS": "Adani Ports",

    "HINDUNILVR": "Hindustan Unilever",

}

def analyze(ticker: str, company\_name: str, sector: str, progress=gr.Progress()):

    """Main analysis function called by Gradio."""

    ticker \= ticker.strip().upper()

    if not ticker:

        return "Please enter a ticker.", {}, \[\], {}, {}, {}, \[\]

    if not company\_name and ticker in EXAMPLE\_COMPANIES:

        company\_name \= EXAMPLE\_COMPANIES\[ticker\]

    elif not company\_name:

        company\_name \= ticker

    progress(0.05, desc="Initiating analysis...")

    try:

        progress(0.15, desc="Fetching data & running agents in parallel...")

        response \= requests.post(

            f"{API\_URL}/analyze",

            json={"ticker": ticker, "company\_name": company\_name, "sector": sector},

            timeout=120,

        )

        response.raise\_for\_status()

        data \= response.json()

    except requests.Timeout:

        return "⚠️ Analysis timed out (\>120s). Please try again.", {}, \[\], {}, {}, {}, \["Timeout error"\]

    except requests.ConnectionError:

        return "⚠️ Cannot connect to FinSight AI backend. Is the server running?\\n\`\`\`\\nuvicorn main:app \--port 8000\\n\`\`\`", {}, \[\], {}, {}, {}, \["Connection error"\]

    except Exception as e:

        return f"⚠️ Error: {str(e)}", {}, \[\], {}, {}, {}, \[str(e)\]

    progress(0.95, desc="Formatting report...")

    report\_md \= data.get("report\_markdown", "No report generated.")

    if data.get("processing\_time\_seconds"):

        report\_md \+= f"\\n\\n---\\n\*⏱ Analysis completed in {data\['processing\_time\_seconds'\]}s\*"

    progress(1.0, desc="Done\!")

    return (

        report\_md,                             \# Investment Report tab

        data.get("filing\_summary", {}),        \# Filing Insights tab

        data.get("ratios", \[\]),                \# Ratios tab

        data.get("news\_sentiment", {}),        \# News tab

        data.get("technical", {}),             \# Technical tab

        data.get("risk\_scorecard", {}),        \# Risk tab

        data.get("errors", \[\]),                \# Errors tab

    )

def load\_example(ticker: str):

    """Load example company into inputs."""

    return ticker, EXAMPLE\_COMPANIES.get(ticker, ""), ""

\# ── UI LAYOUT ─────────────────────────────────────────────────────────────

with gr.Blocks(

    title="FinSight AI",

    theme=gr.themes.Soft(primary\_hue="blue"),

    css="""

    .header-title { font-size: 2rem; font-weight: 700; color: \#1d4ed8; }

    .header-sub { color: \#64748b; font-size: 0.9rem; }

    .risk-high { color: \#dc2626; font-weight: bold; }

    .risk-low  { color: \#16a34a; font-weight: bold; }

    """

) as demo:

    gr.HTML("""

    \<div style='text-align:center; padding: 20px 0 10px 0;'\>

        \<div class='header-title'\>🔍 FinSight AI\</div\>

        \<div class='header-sub'\>Multi-Agent Equity Research Platform for Indian Stocks (NSE/BSE)\</div\>

        \<div class='header-sub' style='color:\#dc2626; margin-top:4px;'\>

            ⚠️ Academic prototype only. Not financial advice. Not SEBI registered.

        \</div\>

    \</div\>

    """)

    with gr.Row():

        with gr.Column(scale=2):

            ticker\_input \= gr.Textbox(

                label="NSE Ticker",

                placeholder="e.g. RELIANCE, HDFCBANK, INFY",

                max\_lines=1,

            )

        with gr.Column(scale=3):

            company\_input \= gr.Textbox(

                label="Company Name",

                placeholder="e.g. Reliance Industries (auto-filled for known tickers)",

                max\_lines=1,

            )

        with gr.Column(scale=2):

            sector\_input \= gr.Dropdown(

                label="Sector (optional)",

                choices=\["", "IT", "BFSI", "Energy", "FMCG", "Pharma", "Auto", "Infra", "Defence", "NBFC", "Diversified"\],

                value="",

            )

    with gr.Row():

        analyze\_btn \= gr.Button("🚀 Analyze", variant="primary", scale=3)

        gr.HTML("\<div style='padding-top:6px; color:\#64748b; font-size:0.8rem;'\>⏱ Takes 15–60 seconds\</div\>")

    \# Example buttons

    gr.HTML("\<p style='color:\#64748b; margin-top:8px; font-size:0.85rem;'\>Quick examples:\</p\>")

    with gr.Row():

        for t in list(EXAMPLE\_COMPANIES.keys())\[:5\]:

            btn \= gr.Button(t, size="sm", variant="secondary")

            btn.click(load\_example, inputs=\[gr.Textbox(value=t, visible=False)\],

                     outputs=\[ticker\_input, company\_input, sector\_input\])

    \# Output tabs

    with gr.Tabs():

        with gr.Tab("📋 Investment Report"):

            report\_out \= gr.Markdown(label="Report")

        with gr.Tab("📄 Filing Insights"):

            gr.HTML("\<p style='color:\#64748b; font-size:0.85rem;'\>Extracted from actual company annual reports via RAG\</p\>")

            filing\_out \= gr.JSON(label="Filing Analysis")

        with gr.Tab("📊 Financial Ratios"):

            gr.HTML("\<p style='color:\#64748b; font-size:0.85rem;'\>Computed from live yfinance data, benchmarked vs sector medians\</p\>")

            ratios\_out \= gr.JSON(label="Ratio Analysis")

        with gr.Tab("📰 News Intelligence"):

            gr.HTML("\<p style='color:\#64748b; font-size:0.85rem;'\>Sentiment analysis of last 20 Google News articles\</p\>")

            news\_out \= gr.JSON(label="News Analysis")

        with gr.Tab("📈 Technical Analysis"):

            gr.HTML("\<p style='color:\#64748b; font-size:0.85rem;'\>RSI, MACD, SMA from 1-year OHLCV data via pandas\_ta\</p\>")

            technical\_out \= gr.JSON(label="Technical Indicators")

        with gr.Tab("⚠️ Risk Scorecard"):

            gr.HTML("\<p style='color:\#64748b; font-size:0.85rem;'\>Beta, VaR, Altman Z-Score, Promoter Pledge and other risk flags\</p\>")

            risk\_out \= gr.JSON(label="Risk Assessment")

        with gr.Tab("🔧 Debug / Errors"):

            errors\_out \= gr.JSON(label="Pipeline Errors (non-fatal)")

    \# Wire up button

    analyze\_btn.click(

        fn=analyze,

        inputs=\[ticker\_input, company\_input, sector\_input\],

        outputs=\[report\_out, filing\_out, ratios\_out, news\_out, technical\_out, risk\_out, errors\_out\],

        show\_progress="full",

    )

    \# Enter key also triggers analysis

    ticker\_input.submit(

        fn=analyze,

        inputs=\[ticker\_input, company\_input, sector\_input\],

        outputs=\[report\_out, filing\_out, ratios\_out, news\_out, technical\_out, risk\_out, errors\_out\],

    )

if \_\_name\_\_ \== "\_\_main\_\_":

    demo.launch(server\_name="0.0.0.0", server\_port=7860, share=False)

---

## 11\. Phase 7 — Evaluation (RAGAS)

### 11.1 `eval/ragas_eval.py`

"""

RAGAS evaluation script for Filing Analyst RAG quality.

Run: python eval/ragas\_eval.py

Measures Faithfulness, Answer Relevancy, Context Recall.

Target: Faithfulness \>= 0.80

"""

import json

import logging

from pathlib import Path

from dotenv import load\_dotenv

load\_dotenv()

logging.basicConfig(level=logging.INFO)

logger \= logging.getLogger(\_\_name\_\_)

def load\_test\_questions() \-\> list\[dict\]:

    """Load the 15-question test set."""

    with open("eval/test\_questions.json") as f:

        return json.load(f)

def run\_ragas\_evaluation():

    from ragas import evaluate

    from ragas.metrics import faithfulness, answer\_relevancy, context\_recall

    from datasets import Dataset

    from backend.rag.retriever import retrieve\_for\_query

    from backend.agents.filing\_analyst import run\_filing\_analyst

    from backend.rag.indexer import ingest\_all\_companies

    \# Ensure all companies are indexed

    logger.info("Checking RAG index...")

    ingest\_all\_companies(force\_reindex=False)

    test\_data \= load\_test\_questions()

    questions \= \[\]

    answers \= \[\]

    contexts \= \[\]

    ground\_truths \= \[\]

    logger.info(f"Evaluating {len(test\_data)} questions...")

    for item in test\_data:

        company \= item\["company"\]

        question \= item\["question"\]

        reference \= item\["ground\_truth"\]

        \# Retrieve context chunks

        retrieved \= retrieve\_for\_query(company, question, top\_k=5)

        if not retrieved:

            logger.warning(f"No chunks retrieved for: {question\[:50\]}")

            continue

        \# Generate answer using Filing Analyst

        from langchain\_openai import ChatOpenAI

        llm \= ChatOpenAI(model="gpt-4o-mini", temperature=0)

        context\_text \= "\\n\\n".join(retrieved\[:5\])

        prompt \= f"Based only on the following excerpts from {company}'s filings:\\n\\n{context\_text}\\n\\nAnswer: {question}"

        answer \= llm.invoke(prompt).content.strip()

        questions.append(question)

        answers.append(answer)

        contexts.append(retrieved\[:5\])

        ground\_truths.append(reference)

    if not questions:

        logger.error("No questions could be evaluated. Ensure companies are indexed.")

        return

    dataset \= Dataset.from\_dict({

        "question": questions,

        "answer": answers,

        "contexts": contexts,

        "ground\_truth": ground\_truths,

    })

    logger.info("Running RAGAS evaluation...")

    result \= evaluate(

        dataset=dataset,

        metrics=\[faithfulness, answer\_relevancy, context\_recall\],

    )

    print("\\n" \+ "="\*50)

    print("RAGAS EVALUATION RESULTS")

    print("="\*50)

    print(f"Faithfulness:      {result\['faithfulness'\]:.3f}  (target: \>= 0.80)")

    print(f"Answer Relevancy:  {result\['answer\_relevancy'\]:.3f}  (target: \>= 0.75)")

    print(f"Context Recall:    {result\['context\_recall'\]:.3f}  (target: \>= 0.70)")

    print("="\*50)

    \# Save results

    output\_path \= Path("eval/ragas\_results.json")

    with open(output\_path, "w") as f:

        json.dump({

            "faithfulness": result\["faithfulness"\],

            "answer\_relevancy": result\["answer\_relevancy"\],

            "context\_recall": result\["context\_recall"\],

            "n\_questions": len(questions),

        }, f, indent=2)

    logger.info(f"Results saved to {output\_path}")

    return result

if \_\_name\_\_ \== "\_\_main\_\_":

    run\_ragas\_evaluation()

### 11.2 `eval/test_questions.json`

\[

  {"company": "HDFCBANK", "question": "What was HDFC Bank's net profit in FY2024?", "ground\_truth": "HDFC Bank reported a net profit of approximately ₹60,812 crore in FY2024."},

  {"company": "HDFCBANK", "question": "What was HDFC Bank's net interest margin in FY2024?", "ground\_truth": "HDFC Bank's net interest margin was approximately 3.4-3.6% in FY2024."},

  {"company": "HDFCBANK", "question": "What are the key risks mentioned in HDFC Bank's annual report?", "ground\_truth": "Key risks include credit risk, interest rate risk, regulatory compliance risk, and concentration risk in loan book."},

  {"company": "RELIANCE", "question": "What was Reliance Industries' revenue in FY2024?", "ground\_truth": "Reliance Industries' consolidated revenue was approximately ₹9,01,386 crore in FY2024."},

  {"company": "RELIANCE", "question": "What segments does Reliance Industries operate in?", "ground\_truth": "Reliance operates in Oil to Chemicals (O2C), Jio Platforms (telecom/digital), Retail, and Oil & Gas exploration."},

  {"company": "RELIANCE", "question": "What was Reliance's capital expenditure guidance for FY2025?", "ground\_truth": "Reliance indicated continued capex in Jio 5G rollout, retail expansion, and new energy businesses."},

  {"company": "INFY", "question": "What was Infosys's revenue growth in FY2024?", "ground\_truth": "Infosys reported revenue of approximately $18.5 billion in FY2024 with modest growth amid muted discretionary spending."},

  {"company": "INFY", "question": "What is Infosys's headcount utilization mentioned in annual report?", "ground\_truth": "Infosys maintained utilization around 80-82% excluding trainees during FY2024."},

  {"company": "INFY", "question": "What are the main risk factors mentioned by Infosys?", "ground\_truth": "Key risks include global macro slowdown reducing IT spending, currency risk, talent retention, and geopolitical uncertainties."},

  {"company": "TCS", "question": "What was TCS's revenue in FY2024?", "ground\_truth": "TCS reported consolidated revenue of approximately ₹2,40,893 crore in FY2024."},

  {"company": "TCS", "question": "What was TCS's order book position in FY2024?", "ground\_truth": "TCS reported a strong order book with total contract value wins of approximately $42.7 billion for FY2024."},

  {"company": "WIPRO", "question": "What strategic initiatives did Wipro mention for FY2024?", "ground\_truth": "Wipro focused on AI-led services, strategic acquisitions, and its consulting arm Wipro FullStride Cloud under CEO Thierry Delaporte."},

  {"company": "BAJFINANCE", "question": "What was Bajaj Finance's AUM in FY2024?", "ground\_truth": "Bajaj Finance's Assets Under Management exceeded ₹3,30,615 crore as of March 2024."},

  {"company": "BAJFINANCE", "question": "What was Bajaj Finance's gross NPA ratio in FY2024?", "ground\_truth": "Bajaj Finance maintained a Gross NPA ratio of approximately 0.85-0.95% in FY2024."},

  {"company": "HINDUNILVR", "question": "What was Hindustan Unilever's volume growth in FY2024?", "ground\_truth": "HUL reported modest volume growth with pressure from rural demand slowdown and competitive intensity from regional players."}

\]

---

## 12\. Static Data Files

### 12.1 `static/sector_medians.json`

{

  "IT": {

    "pe": 25.0,

    "pb": 6.5,

    "roe": 22.0,

    "roce": 28.0,

    "de\_ratio": 0.05,

    "operating\_margin": 20.0,

    "net\_margin": 15.0,

    "current\_ratio": 2.5,

    "sales\_cagr\_3y": 12.0,

    "description": "Indian IT Services sector — Infosys, TCS, Wipro, HCL Tech, Tech Mahindra"

  },

  "BFSI": {

    "pe": 18.0,

    "pb": 2.8,

    "roe": 14.0,

    "roce": null,

    "de\_ratio": null,

    "operating\_margin": null,

    "net\_margin": 18.0,

    "nim": 3.5,

    "gnpa": 2.5,

    "description": "Banking, Financial Services & Insurance — HDFC Bank, ICICI Bank, SBI, Kotak"

  },

  "FMCG": {

    "pe": 52.0,

    "pb": 12.0,

    "roe": 50.0,

    "roce": 60.0,

    "de\_ratio": 0.1,

    "operating\_margin": 18.0,

    "net\_margin": 12.0,

    "current\_ratio": 1.8,

    "sales\_cagr\_3y": 9.0,

    "description": "Fast Moving Consumer Goods — HUL, ITC, Nestle, Dabur, Britannia"

  },

  "Pharma": {

    "pe": 28.0,

    "pb": 4.5,

    "roe": 15.0,

    "roce": 18.0,

    "de\_ratio": 0.3,

    "operating\_margin": 18.0,

    "net\_margin": 11.0,

    "current\_ratio": 2.0,

    "sales\_cagr\_3y": 11.0,

    "description": "Pharmaceuticals — Sun Pharma, Dr Reddy's, Cipla, Divi's"

  },

  "Auto": {

    "pe": 22.0,

    "pb": 4.0,

    "roe": 18.0,

    "roce": 22.0,

    "de\_ratio": 0.5,

    "operating\_margin": 12.0,

    "net\_margin": 7.0,

    "current\_ratio": 1.2,

    "sales\_cagr\_3y": 14.0,

    "description": "Automobiles — Maruti, Tata Motors, M\&M, Hero MotoCorp"

  },

  "Energy": {

    "pe": 12.0,

    "pb": 1.8,

    "roe": 14.0,

    "roce": 16.0,

    "de\_ratio": 0.8,

    "operating\_margin": 15.0,

    "net\_margin": 8.0,

    "current\_ratio": 1.0,

    "sales\_cagr\_3y": 10.0,

    "description": "Oil & Gas, Power — Reliance, ONGC, NTPC, Tata Power, BPCL"

  },

  "Infra": {

    "pe": 35.0,

    "pb": 4.5,

    "roe": 12.0,

    "roce": 14.0,

    "de\_ratio": 1.5,

    "operating\_margin": 20.0,

    "net\_margin": 6.0,

    "current\_ratio": 1.1,

    "sales\_cagr\_3y": 18.0,

    "description": "Infrastructure, Ports, Construction — Adani Ports, L\&T, DLF"

  },

  "Defence": {

    "pe": 40.0,

    "pb": 6.0,

    "roe": 16.0,

    "roce": 20.0,

    "de\_ratio": 0.1,

    "operating\_margin": 15.0,

    "net\_margin": 10.0,

    "current\_ratio": 2.5,

    "sales\_cagr\_3y": 20.0,

    "description": "Defence PSUs and private — BEL, HAL, Cochin Shipyard, MTAR"

  },

  "NBFC": {

    "pe": 22.0,

    "pb": 4.0,

    "roe": 18.0,

    "roce": null,

    "de\_ratio": 3.5,

    "operating\_margin": null,

    "net\_margin": 20.0,

    "description": "Non-Banking Financial Companies — Bajaj Finance, Muthoot Finance, Cholamandalam"

  },

  "Diversified": {

    "pe": 22.0,

    "pb": 3.5,

    "roe": 15.0,

    "roce": 18.0,

    "de\_ratio": 0.6,

    "operating\_margin": 15.0,

    "net\_margin": 9.0,

    "current\_ratio": 1.5,

    "sales\_cagr\_3y": 12.0,

    "description": "Default — used when sector is unknown or diversified conglomerate"

  }

}

---

## 13\. Build Order for Claude Code

**Follow this exact sequence. Each phase must work before moving to the next.**

PHASE 0 — PROJECT SCAFFOLD

□ Create folder structure from Section 3

□ Create .env, .env.example, .gitignore

□ Create requirements.txt

□ Run: pip install \-r requirements.txt

□ Create static/sector\_medians.json (copy from Section 12.1)

□ Create all \_\_init\_\_.py files

PHASE 1 — DATA PIPELINE (test each independently)

□ Build backend/utils/cache.py

□ Build backend/utils/sector\_data.py

□ Build backend/tools/financial\_tools.py

  → Test: python \-c "from backend.tools.financial\_tools import get\_stock\_info; print(get\_stock\_info.func('RELIANCE.NS'))"

□ Build backend/tools/news\_tools.py

  → Test: python \-c "from backend.tools.news\_tools import fetch\_google\_news; print(fetch\_google\_news('Reliance Industries')\[:2\])"

□ Build backend/tools/technical\_tools.py

  → Test: python \-c "from backend.tools.technical\_tools import compute\_technical\_indicators; ..."

□ Build backend/utils/validators.py

PHASE 2 — RAG PIPELINE

□ Download PDF filings: create data/filings/ structure, add at least 2 PDFs for HDFCBANK and RELIANCE

□ Build backend/rag/indexer.py

  → Test: python \-c "from backend.rag.indexer import ingest\_company\_filings; print(ingest\_company\_filings('HDFCBANK'))"

□ Build backend/rag/retriever.py

  → Test: python \-c "from backend.rag.retriever import retrieve\_for\_query; print(retrieve\_for\_query('HDFCBANK', 'net profit FY24')\[:1\])"

PHASE 3 — INDIVIDUAL AGENTS (build and test each one)

□ Build backend/agents/filing\_analyst.py

  → Test: python \-c "from backend.agents.filing\_analyst import run\_filing\_analyst; print(run\_filing\_analyst('HDFCBANK.NS', 'HDFC Bank'))"

□ Build backend/agents/ratio\_cruncher.py

  → Test: python \-c "from backend.agents.ratio\_cruncher import run\_ratio\_cruncher; print(run\_ratio\_cruncher('INFY.NS', 'Infosys', 'IT'))"

□ Build backend/agents/news\_sentinel.py

  → Test: python \-c "from backend.agents.news\_sentinel import run\_news\_sentinel; print(run\_news\_sentinel('RELIANCE.NS', 'Reliance Industries'))"

□ Build backend/agents/technical\_analyst.py

  → Test: python \-c "from backend.agents.technical\_analyst import run\_technical\_analyst; print(run\_technical\_analyst('TCS.NS', 'TCS'))"

□ Build backend/risk/metrics.py

□ Build backend/agents/risk\_assessor.py

  → Test with dummy state dicts

□ Build backend/agents/report\_writer.py

  → Test with dummy inputs from above agents

PHASE 4 — LANGGRAPH ORCHESTRATION

□ Build backend/graph/state.py

□ Build backend/graph/workflow.py

  → Test end-to-end: python \-c "from backend.graph.workflow import run\_analysis; r \= run\_analysis('INFY.NS', 'Infosys', 'IT'); print(r\['final\_report'\]\['status'\])"

PHASE 5 — FASTAPI BACKEND

□ Build main.py

  → Run: uvicorn main:app \--port 8000 \--reload

  → Test: curl http://localhost:8000/health

  → Test: curl \-X POST http://localhost:8000/analyze \-H "Content-Type: application/json" \-d '{"ticker":"INFY","company\_name":"Infosys"}'

PHASE 6 — GRADIO FRONTEND

□ Build frontend/app.py

  → Run: python frontend/app.py

  → Open browser: http://localhost:7860

  → Test with INFY ticker

PHASE 7 — EVALUATION

□ Download enough PDF filings for 15 test questions

□ Create eval/test\_questions.json (copy from Section 11.2)

□ Build eval/ragas\_eval.py

  → Run: python eval/ragas\_eval.py

  → Target: Faithfulness \>= 0.80

---

## 14\. Testing Checklist

### Unit tests `tests/test_tools.py`

import pytest

from backend.tools.financial\_tools import get\_stock\_info, get\_price\_history, get\_sector\_median

def test\_get\_stock\_info\_reliance():

    result \= get\_stock\_info.func("RELIANCE.NS")

    assert "error" not in result

    assert result.get("current\_price") is not None

    assert result.get("pe\_ratio") is not None

def test\_get\_stock\_info\_adds\_ns\_suffix():

    result \= get\_stock\_info.func("RELIANCE")  \# No .NS suffix

    assert "error" not in result

def test\_get\_price\_history\_returns\_lists():

    result \= get\_price\_history.func("INFY.NS", "6mo")

    assert isinstance(result.get("close"), list)

    assert len(result\["close"\]) \> 100

def test\_sector\_median\_it():

    result \= get\_sector\_median.func("IT")

    assert "medians" in result

    assert result\["medians"\].get("pe") is not None

def test\_sector\_median\_fallback():

    result \= get\_sector\_median.func("UnknownSector")

    assert "medians" in result  \# Should return Diversified

### Integration smoke test

\# tests/test\_graph.py

import pytest

from backend.graph.workflow import run\_analysis

def test\_full\_pipeline\_infy():

    """End-to-end pipeline test for Infosys."""

    state \= run\_analysis("INFY.NS", "Infosys", "IT")

    

    assert state\["ratio\_output"\] is not None

    assert state\["news\_output"\] is not None

    assert state\["technical\_output"\] is not None

    assert state\["risk\_output"\] is not None

    assert state\["final\_report"\] is not None

    

    \# Report should have content even if some agents failed

    report \= state\["final\_report"\]

    assert report.get("status") in \["success", "partial\_success"\]

    

    \# Ratio output should have ratios list

    assert isinstance(state\["ratio\_output"\].get("ratios"), list)

    

    \# Risk output should have a risk level

    assert state\["risk\_output"\].get("risk\_level") in \[

        "LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH", "VERY HIGH", "UNKNOWN"

    \]

---

## IMPORTANT NOTES FOR CLAUDE CODE

1. **Never scrape nseindia.com or bseindia.com** — use yfinance for price data, direct PDF download for filings  
2. **All agent functions must have non-fatal error handling** — wrap in try/except and return error dict, never raise  
3. **Every tool that calls an external API must use the caching layer** — import from `backend.utils.cache`  
4. **Use `.NS` suffix for all NSE tickers** — the `_ensure_ns()` helper in financial\_tools handles this  
5. **GPT-4o-mini for all agents except Report Writer** — Report Writer uses GPT-4o  
6. **ChromaDB persists to disk** — do not create a new client on every function call; use the persistent path  
7. **FinSightState fields are Optional\[dict\]** — always check for None before accessing fields in downstream agents  
8. **JSON parsing from LLM responses** — always strip markdown code fences before parsing; LLMs often wrap JSON in \`\`\`json  
9. **The four specialist agents (filing, ratio, news, technical) run in PARALLEL** — they must not depend on each other's outputs  
10. **Risk Assessor and Report Writer run SEQUENTIALLY after all four agents** — this is enforced by the LangGraph edges

---

*FinSight AI — VIT Chennai | B.Tech CSE with AI/ML | GenAI & Agentic AI Course Project 2025–26*  
