# FinSight AI - Quick Start Commands

## Complete Step-by-Step Commands

### 1. Navigate to Project
```bash
cd E:/PJT1/finsight
```

### 2. Activate Virtual Environment
```bash
.venv/Scripts/activate
```

### 3. Verify Environment Variables
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('GOOGLE_API_KEY:', 'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET')"
```

### 4. (Optional) Index Company Filings
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from backend.rag import ingest_all_companies; result = ingest_all_companies(); print(f'Indexed {result[\"companies_processed\"]} companies')"
```

### 5. Start the Application
```bash
python main.py
```

### 6. Open Browser
- Frontend: http://localhost:7860
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## One-Line Startup
```bash
cd E:/PJT1/finsight && .venv/Scripts/activate && python main.py
```

## Testing Commands

### Run All Tests
```bash
pytest
```

### Test RAG System
```bash
python test_simplified_rag.py
```

### Test Individual Components
```bash
# Test financial tools
pytest tests/smoke_test_financial_tools.py

# Test filing analyst  
pytest tests/smoke_test_filing_analyst.py

# Test RAG indexer
pytest tests/test_rag_indexer.py
```

## Useful Commands

### Check Backend Health
```bash
curl http://localhost:8000/health
```

### Analyze a Stock
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"ticker\": \"RELIANCE.NS\", \"company_name\": \"Reliance Industries\"}"
```

### Reindex All Companies
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from backend.rag import ingest_all_companies; ingest_all_companies(force_reindex=True)"
```

### Check Installed Packages
```bash
pip list | grep -E "fastapi|gradio|google|faiss|langchain"
```

## Troubleshooting

### Port Already in Use
```bash
# Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

### Reinstall Dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

### Clear Vector Store Cache
```bash
rm -rf data/vector_store/*
```

### View Logs
```bash
# Backend logs are in the console
# Set log level in .env: LOG_LEVEL=DEBUG
```

---

**All systems are ready! Run `python main.py` to start.**
