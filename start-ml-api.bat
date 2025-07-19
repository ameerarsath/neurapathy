@echo off
REM ML API Startup Script for Windows
echo 🚀 Starting ML API for Smart Shoe Application...

REM Check if we're in the right directory
if not exist "ml-models" (
    echo ❌ Error: ml-models directory not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Navigate to ML models directory
cd ml-models

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo 📦 Checking Python dependencies...
if exist "requirements.txt" (
    echo Installing/updating requirements...
    pip install -r requirements.txt
) else (
    echo ⚠️  Warning: requirements.txt not found. Installing common dependencies...
    pip install fastapi uvicorn joblib numpy pandas scikit-learn mlflow
)

REM Check which API to start (prefer minimal for reliability)
if exist "src\deployment\minimal_api.py" (
    echo 🔧 Starting Minimal ML API (no dependencies required)...
    python src\deployment\minimal_api.py
) else if exist "src\deployment\enhanced_api.py" (
    echo 🔧 Starting Enhanced ML API...
    python src\deployment\enhanced_api.py
) else if exist "src\deployment\api_integration.py" (
    echo 🔧 Starting Basic ML API...
    python src\deployment\api_integration.py
) else (
    echo ❌ Error: ML API files not found. Please check the ml-models directory structure.
    pause
    exit /b 1
)

pause