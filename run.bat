@echo off
REM FinSight AI - Windows Startup Script

echo ==================================================
echo Starting FinSight AI
echo ==================================================

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check for .env file
if not exist .env (
    echo ERROR: .env file not found!
    echo Please create a .env file with your API keys.
    echo See .env.example for reference.
    pause
    exit /b 1
)

echo.
echo Starting FastAPI Backend on http://localhost:8000
echo Starting Gradio Frontend on http://localhost:7860
echo.
echo Press Ctrl+C to stop both servers
echo.

REM Start the application
python main.py

pause
