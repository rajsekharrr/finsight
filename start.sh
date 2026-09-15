#!/bin/bash
# FinSight AI - Startup Script
# This script starts both the FastAPI backend and Gradio frontend

echo "=================================================="
echo "Starting FinSight AI"
echo "=================================================="

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source .venv/Scripts/activate || source .venv/bin/activate
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your API keys."
    echo "See .env.example for reference."
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Check for required API keys
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "WARNING: GOOGLE_API_KEY not set in .env"
    echo "RAG functionality will not work without it."
fi

echo ""
echo "Starting FastAPI Backend on http://localhost:8000"
echo "Starting Gradio Frontend on http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start both servers
python main.py
