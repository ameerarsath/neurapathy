from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram
import json_logging
from pathlib import Path
import mlflow
from mlflow.pyfunc import load_model

# Initialize logging
json_logging.init_fastapi()
logger = logging.getLogger("ml_api")

# Initialize metrics
PREDICTION_COUNTER = Counter('prediction_requests_total', 'Total prediction requests')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

# Initialize FastAPI app
app = FastAPI(
    title="Smart Shoe ML API",
    description="API for diabetic neuropathy prediction and monitoring",
    version="1.0.0"
)

# Load MLflow model
MODEL_PATH = Path(__file__).parent.parent.parent / "experiments" / "mlruns"
MLFLOW_TRACKING_URI = f"file://{MODEL_PATH}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_production_model():
    """Load the latest production model from MLflow."""
    try:
        # Get the latest version of the registered model
        model_name = "smart_shoe_model"
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Production"])[0]
        model_uri = f"models:/{model_name}/Production"
        return load_model(model_uri)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

class PredictionInput(BaseModel):
    patient_id: str
    timestamp: datetime
    measurements: Dict[str, float]
    context: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

class PredictionResponse(BaseModel):
    patient_id: str
    timestamp: datetime
    prediction: float
    risk_level: str
    confidence: float

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    model = load_production_model()
    logger.info("Model loaded successfully")

@app.post("/predict/risk", response_model=PredictionResponse)
async def predict_risk(input_data: PredictionInput):
    """
    Predict risk level using the MLflow model.
    """
    PREDICTION_COUNTER.inc()
    
    try:
        with PREDICTION_LATENCY.time():
            # Prepare input data
            features = np.array([list(input_data.measurements.values())])
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Calculate risk level and confidence
            risk_level = "HIGH" if prediction > 0.7 else "MEDIUM" if prediction > 0.3 else "LOW"
            confidence = float(abs(prediction - 0.5) * 2)  # Simple confidence calculation
            
            return PredictionResponse(
                patient_id=input_data.patient_id,
                timestamp=input_data.timestamp,
                prediction=float(prediction),
                risk_level=risk_level,
                confidence=confidence
            )
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 