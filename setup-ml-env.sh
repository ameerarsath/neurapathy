#!/bin/bash

# ML Environment Setup Script for Linux/Mac
echo "🔧 Setting up ML Environment for Smart Shoe Application..."

# Check if we're in the right directory
if [ ! -d "ml-models" ]; then
    echo "❌ Error: ml-models directory not found. Please run this script from the project root directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "📦 Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel

echo "📦 Installing core dependencies one by one..."
python3 -m pip install fastapi
python3 -m pip install uvicorn
python3 -m pip install pydantic
python3 -m pip install requests
python3 -m pip install python-multipart

echo "📦 Installing ML dependencies..."
python3 -m pip install numpy
python3 -m pip install pandas
python3 -m pip install scikit-learn
python3 -m pip install joblib

echo "📦 Installing optional dependencies..."
python3 -m pip install mlflow || echo "⚠️ MLflow installation failed, continuing..."

echo "✅ Dependencies installed successfully!"
echo "🚀 Now you can run: ./start-ml-api.sh"