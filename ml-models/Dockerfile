# Dockerfile for ML API Service
# Production-optimized Python ML API

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlapi

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src
COPY trained-models ./trained-models
COPY data ./data

# Create necessary directories
RUN mkdir -p /app/logs && \
    mkdir -p /app/experiments && \
    chown -R mlapi:mlapi /app

# Switch to non-root user
USER mlapi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/trained-models
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["python", "-m", "uvicorn", "src.deployment.enhanced_api:app", "--host", "0.0.0.0", "--port", "8000"]