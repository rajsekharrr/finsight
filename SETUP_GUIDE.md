# FinSight AI - Complete Setup and Run Guide

## ✅ Stack Simplification Complete

The RAG system has been successfully simplified:

### Removed Dependencies
- ❌ llama-index (all packages)
- ❌ chromadb
- ❌ OpenAI embeddings

### New Simple Stack
- ✅ google-generativeai for embeddings
- ✅ faiss-cpu for vector store
- ✅ pypdf for PDF processing
- ✅ Direct Python code for chunking and retrieval

## 📋 Prerequisites

1. **Python 3.11+** installed
2. **Virtual environment** activated
3. **API Keys** configured in `.env` file:
   - `GOOGLE_API_KEY` (required for RAG embeddings)
   - `OPENAI_API_KEY` (optional, for OpenAI models)

## 🚀 Quick Start

### Option 1: All-in-One Command
```bash
cd E:/PJT1/finsight
.venv/Scripts/activate
python main.py
```

### Option 2: Separate Backend and Frontend

**Terminal 1 - Backend:**
```bash
cd E:/PJT1/finsight
.venv/Scripts/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd E:/PJT1/finsight
.venv/Scripts/activate
python frontend/app.py
```

## 📊 Application URLs

- **Frontend (Gradio UI)**: http://localhost:7860
- **Backend API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🗂️ RAG System Setup

### Index Company Filings

```bash
cd E:/PJT1/finsight
python -c "
from dotenv import load_dotenv
load_dotenv()
from backend.rag import ingest_all_companies

result = ingest_all_companies(force_reindex=False)
print(result)
"
```

### Current Indexed Companies
- ✅ HDFCBANK (2 chunks)
- ✅ RELIANCE (2 chunks)

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Test Simplified RAG System
```bash
python test_simplified_rag.py
```

### Test Specific Components
```bash
# Test financial tools
pytest tests/smoke_test_financial_tools.py

# Test RAG indexer
pytest tests/test_rag_indexer.py

# Test filing analyst
pytest tests/smoke_test_filing_analyst.py
```

## 📁 Project Structure

```
finsight/
├── backend/
│   ├── agents/          # Six specialist agents
│   ├── graph/           # LangGraph workflow
│   ├── rag/             # Simplified RAG system
│   │   ├── document_processor.py  # PDF loading & chunking
│   │   ├── gemini_embedding.py    # Gemini embeddings
│   │   ├── vector_store.py        # FAISS wrapper
│   │   ├── indexer.py             # Indexing PDFs
│   │   └── retriever.py           # Querying vectors
│   ├── tools/           # Financial, news, technical tools
│   ├── utils/           # Cache, validators, sector data
│   └── risk/            # Risk metrics
├── frontend/
│   └── app.py           # Gradio interface
├── data/
│   ├── filings/         # Company PDF filings
│   │   ├── HDFCBANK/
│   │   └── RELIANCE/
│   └── vector_store/    # FAISS indexes
├── tests/               # Test suite
├── main.py              # FastAPI backend entry point
├── requirements.txt     # Python dependencies
└── .env                 # API keys (not in git)
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional
OPENAI_API_KEY=your_openai_key_here

# App Settings
APP_HOST=0.0.0.0
APP_PORT=8000
CACHE_TTL_HOURS=6
LOG_LEVEL=INFO
```

## 📝 API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Analyze Stock
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE.NS", "company_name": "Reliance Industries", "sector": "Energy"}'
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process
taskkill /PID <pid> /F
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### RAG Not Working
1. Check GOOGLE_API_KEY is set: `echo $GOOGLE_API_KEY`
2. Verify embeddings model: `models/gemini-embedding-001`
3. Re-index filings: `ingest_all_companies(force_reindex=True)`

### Import Errors
```bash
# Verify Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Reinstall packages
pip install --force-reinstall google-generativeai
```

## ✨ Key Features

1. **Multi-Agent System**: Six specialized agents for comprehensive analysis
2. **RAG-Powered**: Retrieves relevant information from company filings
3. **Real-Time Data**: Fetches live stock data via yfinance
4. **News Sentiment**: Analyzes recent news for sentiment
5. **Technical Analysis**: Calculates indicators and signals
6. **Risk Assessment**: Evaluates market and financial risks
7. **Comprehensive Reports**: Synthesizes all findings into actionable insights

## 📊 Test Results

```
✓ Text Chunking          PASS
✓ Gemini Embeddings      PASS (3072 dimensions)
✓ FAISS Vector Store     PASS
✓ End-to-End RAG         PASS
✓ PDF Processing         PASS
✓ Retrieval              PASS (2 chunks retrieved)
```

## 🎯 Next Steps

1. **Add More Companies**: Place PDF filings in `data/filings/<COMPANY>/`
2. **Run Indexing**: Execute `ingest_all_companies()`
3. **Test Analysis**: Try analyzing different stocks through the UI
4. **Monitor Logs**: Check console output for errors
5. **Extend Agents**: Add custom logic to specialist agents

## 📚 Documentation

- FastAPI Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Gradio Interface: http://localhost:7860

## 🔗 Support

For issues or questions:
1. Check logs in console output
2. Review test results: `python test_simplified_rag.py`
3. Verify API keys are set correctly
4. Ensure all dependencies are installed

---

**Status**: ✅ All systems operational
**Last Updated**: 2026-09-15
**Version**: 0.1.0
