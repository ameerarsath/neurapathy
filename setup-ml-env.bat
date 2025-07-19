@echo off
REM ML Environment Setup Script for Windows
echo 🔧 Setting up ML Environment for Smart Shoe Application...

REM Check if we're in the right directory
if not exist "ml-models" (
    echo ❌ Error: ml-models directory not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo 📦 Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel

echo 📦 Installing core dependencies one by one...
python -m pip install fastapi
python -m pip install uvicorn
python -m pip install pydantic
python -m pip install requests
python -m pip install python-multipart

echo 📦 Installing ML dependencies...
python -m pip install numpy
python -m pip install pandas
python -m pip install scikit-learn
python -m pip install joblib

echo 📦 Installing optional dependencies...
python -m pip install mlflow || echo "⚠️ MLflow installation failed, continuing..."

echo ✅ Dependencies installed successfully!
echo 🚀 Now you can run: start-ml-api.bat

pause