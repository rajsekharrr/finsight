# FinSight AI — Multi-Agent Equity Research Platform

**Autonomous AI-powered equity research for Indian stocks**

> **Academic Project**  
> Team: Rajsekhar Singha Roy (23BAI1321) · Shubham Dubey (23BAI1344) · Ayush Achin (23BAI1055)  
> Guide: Khadar Nawas K | VIT Chennai | GenAI & Agentic AI Course Project

---

## What it does

User types a stock ticker (e.g. `RELIANCE` or `HDFCBANK`). FinSight AI runs six specialized AI agents in a LangGraph-orchestrated pipeline and produces a structured investment research report covering:

- **Fundamental analysis** from actual company filing PDFs (via RAG)
- **Financial ratio analysis** with sector benchmarking
- **News sentiment analysis** with theme clustering
- **Technical price analysis** (RSI, MACD, SMA)
- **India-specific risk scorecard** (Beta, VaR, Altman Z-Score, Promoter Pledge)
- **Final synthesized report** with Bull case / Bear case / Confidence score

---

## Tech Stack

- **LLM Framework**: LangChain + LangGraph for agent orchestration
- **Models**: GPT-4o-mini (specialist agents), GPT-4o (final synthesis)
- **RAG**: LlamaIndex + ChromaDB for company filing search
- **Financial Data**: yfinance (NSE/BSE stocks)
- **News**: Google News RSS (via feedparser)
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Evaluation**: RAGAS

---

## Setup

### 1. Clone and navigate
```bash
cd finsight-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 4. Add company filings (PDFs)
Download annual reports and place them in:
```
data/filings/RELIANCE/annual_report_FY24.pdf
data/filings/HDFCBANK/annual_report_FY24.pdf
...
```

### 5. Run the app
```bash
# Backend API
python main.py

# Frontend (in another terminal)
python frontend/app.py
```

Access the Gradio UI at http://localhost:7860

---

## Project Structure

```
finsight-ai/
├── backend/
│   ├── agents/          # Six specialist agents
│   ├── graph/           # LangGraph orchestration
│   ├── tools/           # LangChain tools (yfinance, news, technical)
│   ├── rag/             # PDF indexing & retrieval
│   ├── risk/            # Risk metrics computation
│   └── utils/           # Caching, sector data, validators
├── data/
│   ├── filings/         # Company PDFs (user-provided)
│   ├── screener_cache/  # Pre-downloaded Screener.in data
│   └── cache/           # Runtime API cache
├── frontend/
│   └── app.py           # Gradio interface
├── eval/
│   └── ragas_eval.py    # RAGAS evaluation script
├── tests/               # Unit & integration tests
└── static/
    └── sector_medians.json  # Hardcoded sector benchmarks
```

---

## Design Principles

1. **Parallel execution** — four specialist agents run simultaneously via LangGraph `Send()` API
2. **No hallucination** — financial ratios are computed in code, not inferred by the LLM
3. **Auditability** — every claim traces to a specific agent and data source
4. **India-first** — NSE/BSE filings, `.NS` yfinance tickers, promoter pledge flags, Nifty 50 Beta
5. **Resilient** — non-fatal error handling: one agent failing does not stop the pipeline
6. **Legal** — no scraping of nseindia.com or bseindia.com (Terms of Use restrict this)

---

## Legal & Data Sources

| Source | What | API/Method |
|--------|------|------------|
| BSE/NSE company filings | Annual reports, quarterly results (PDFs) | Manual download + official BSE announcement API |
| yfinance | Price, OHLCV, fundamentals | `yfinance` Python library, `.NS` suffix |
| Google News RSS | Recent news headlines | `feedparser`, no API key needed |
| Screener.in | ROCE, CAGR, Promoter %, Pledge % | Pre-exported Excel, cached CSV |
| Nifty 50 | Market benchmark for Beta | yfinance `^NSEI` ticker |

---

## Disclaimer

This is an academic research prototype built for educational purposes only.

- **Not SEBI-registered**. Not financial advice.
- **Data accuracy not guaranteed**. Do not make investment decisions based on this output.
- **For demonstration purposes only**. No warranty of any kind.

---

## License

Academic project — VIT Chennai GenAI Course 2024
