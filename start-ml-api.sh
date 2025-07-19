#!/bin/bash

# ML API Startup Script
echo "🚀 Starting ML API for Smart Shoe Application..."

# Check if we're in the right directory
if [ ! -d "ml-models" ]; then
    echo "❌ Error: ml-models directory not found. Please run this script from the project root directory."
    exit 1
fi

# Navigate to ML models directory
cd ml-models

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking Python dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Installing/updating requirements..."
    pip3 install -r requirements.txt
else
    echo "⚠️  Warning: requirements.txt not found. Installing common dependencies..."
    pip3 install fastapi uvicorn joblib numpy pandas scikit-learn mlflow
fi

# Check which API to start (prefer minimal for reliability)
if [ -f "src/deployment/minimal_api.py" ]; then
    echo "🔧 Starting Minimal ML API (no dependencies required)..."
    python3 src/deployment/minimal_api.py
elif [ -f "src/deployment/enhanced_api.py" ]; then
    echo "🔧 Starting Enhanced ML API..."
    python3 src/deployment/enhanced_api.py
elif [ -f "src/deployment/api_integration.py" ]; then
    echo "🔧 Starting Basic ML API..."
    python3 src/deployment/api_integration.py
else
    echo "❌ Error: ML API files not found. Please check the ml-models directory structure."
    exit 1
fi