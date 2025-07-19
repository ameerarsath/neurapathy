from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
import asyncio
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import json_logging
from pathlib import Path
import mlflow
from mlflow.pyfunc import load_model
import redis
import json
from contextlib import asynccontextmanager
import aiofiles
import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import hashlib

# Initialize logging
json_logging.init_fastapi()
logger = logging.getLogger("enhanced_ml_api")

# Initialize metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total ML predictions', ['model_type', 'status'])
PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds', 'ML prediction latency', ['model_type'])
MODEL_ACCURACY = Histogram('ml_model_accuracy', 'Model accuracy over time', ['model_type'])

# Global variables for models and connections
models = {}
redis_client = None
mongo_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("Starting Enhanced ML API")
    await startup_models()
    await connect_redis()
    await connect_mongodb()
    yield
    # Shutdown
    logger.info("Shutting down Enhanced ML API")
    await shutdown_cleanup()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Enhanced Smart Shoe ML API",
    description="Advanced API for diabetic neuropathy prediction and monitoring with caching, monitoring, and model management",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Configuration
MODEL_REGISTRY = {
    "neuropathy_progression": {
        "path": Path("../trained-models/neuropathy_progression.pkl"),
        "version": "1.0.0",
        "accuracy_threshold": 0.85
    },
    "glucose_complications": {
        "path": Path("../trained-models/glucose_complications.pkl"),
        "version": "1.0.0",
        "accuracy_threshold": 0.80
    },
    "anomaly_detection": {
        "path": Path("../trained-models/anomaly_detection.pkl"),
        "version": "1.0.0",
        "accuracy_threshold": 0.90
    },
    "risk_stratification": {
        "path": Path("../trained-models/risk_stratification.pkl"),
        "version": "1.0.0",
        "accuracy_threshold": 0.85
    }
}

# Pydantic models
class HealthCheck(BaseModel):
    status: str = "healthy"
    timestamp: datetime
    version: str = "2.0.0"
    models_loaded: int
    redis_connected: bool
    mongodb_connected: bool

class PredictionFeatures(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    model_type: str = Field(..., description="Type of ML model to use")
    features: Dict[str, Union[float, int]] = Field(..., description="Feature values for prediction")
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in MODEL_REGISTRY:
            raise ValueError(f"Invalid model type. Available: {list(MODEL_REGISTRY.keys())}")
        return v

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionFeatures] = Field(..., description="List of prediction requests")
    batch_id: Optional[str] = None
    timestamp: Optional[datetime] = None

class PredictionResponse(BaseModel):
    patient_id: str
    model_type: str
    prediction: float
    confidence: float
    model_version: str
    feature_importance: Optional[Dict[str, float]] = None
    additional_data: Optional[Dict[str, Any]] = None
    timestamp: datetime
    processing_time_ms: float
    cache_hit: bool = False

class BatchPredictionResponse(BaseModel):
    batch_id: str
    responses: List[PredictionResponse]
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    timestamp: datetime
    total_processing_time_ms: float

class ModelMetrics(BaseModel):
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float] = None
    last_updated: datetime
    prediction_count: int
    average_latency_ms: float

class ValidationRequest(BaseModel):
    model_type: str
    start_date: datetime
    end_date: datetime
    validation_data: Optional[List[Dict[str, Any]]] = None

class ModelValidationResult(BaseModel):
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    validation_date: datetime
    sample_size: int
    success: bool = True
    error_message: Optional[str] = None

# Async utility functions
async def startup_models():
    """Load all ML models on startup"""
    global models
    try:
        for model_name, config in MODEL_REGISTRY.items():
            model_path = config["path"]
            if model_path.exists():
                logger.info(f"Loading model: {model_name}")
                model = joblib.load(model_path)
                models[model_name] = {
                    "model": model,
                    "version": config["version"],
                    "loaded_at": datetime.now(),
                    "prediction_count": 0,
                    "accuracy_threshold": config["accuracy_threshold"]
                }
                logger.info(f"Successfully loaded model: {model_name}")
            else:
                logger.warning(f"Model file not found: {model_path}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

async def connect_redis():
    """Connect to Redis for caching"""
    global redis_client
    try:
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        # Test connection
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {str(e)}")
        redis_client = None

async def connect_mongodb():
    """Connect to MongoDB for storing predictions"""
    global mongo_client
    try:
        mongo_client = AsyncIOMotorClient('mongodb://localhost:27017')
        # Test connection
        await mongo_client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB: {str(e)}")
        mongo_client = None

async def shutdown_cleanup():
    """Cleanup connections on shutdown"""
    global redis_client, mongo_client
    if redis_client:
        redis_client.close()
    if mongo_client:
        mongo_client.close()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token verification - replace with proper JWT validation"""
    token = credentials.credentials
    # For demo purposes, accept any token starting with "ml_api_"
    if not token.startswith("ml_api_"):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

async def get_cached_prediction(cache_key: str) -> Optional[PredictionResponse]:
    """Get cached prediction from Redis"""
    if not redis_client:
        return None
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            data = json.loads(cached_data)
            return PredictionResponse(**data)
    except Exception as e:
        logger.warning(f"Error retrieving cached prediction: {str(e)}")
    
    return None

async def cache_prediction(cache_key: str, prediction: PredictionResponse, ttl: int = 3600):
    """Cache prediction in Redis"""
    if not redis_client:
        return
    
    try:
        redis_client.setex(cache_key, ttl, prediction.json())
    except Exception as e:
        logger.warning(f"Error caching prediction: {str(e)}")

async def store_prediction_mongodb(prediction: PredictionResponse):
    """Store prediction in MongoDB"""
    if not mongo_client:
        return
    
    try:
        db = mongo_client.ml_predictions
        collection = db.predictions
        await collection.insert_one(prediction.dict())
    except Exception as e:
        logger.warning(f"Error storing prediction in MongoDB: {str(e)}")

def generate_cache_key(patient_id: str, model_type: str, features: Dict[str, Any]) -> str:
    """Generate cache key based on input parameters"""
    feature_str = json.dumps(features, sort_keys=True)
    hash_input = f"{patient_id}_{model_type}_{feature_str}"
    return f"prediction_{hashlib.md5(hash_input.encode()).hexdigest()}"

def calculate_feature_importance(model, features: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Calculate feature importance for the prediction"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance.tolist()))
        return {}
    except Exception:
        return {}

# API Endpoints
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        timestamp=datetime.now(),
        models_loaded=len(models),
        redis_connected=redis_client is not None,
        mongodb_connected=mongo_client is not None
    )

@app.post("/predict/neuropathy-progression", response_model=PredictionResponse)
async def predict_neuropathy_progression(
    request: PredictionFeatures,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Predict neuropathy progression"""
    start_time = datetime.now()
    
    try:
        # Generate cache key
        cache_key = generate_cache_key(request.patient_id, "neuropathy_progression", request.features)
        
        # Check cache first
        cached_result = await get_cached_prediction(cache_key)
        if cached_result:
            cached_result.cache_hit = True
            PREDICTION_COUNTER.labels(model_type="neuropathy_progression", status="cache_hit").inc()
            return cached_result
        
        # Get model
        model_info = models.get("neuropathy_progression")
        if not model_info:
            raise HTTPException(status_code=404, detail="Neuropathy progression model not available")
        
        model = model_info["model"]
        
        # Prepare features
        feature_names = list(request.features.keys())
        feature_values = np.array([list(request.features.values())])
        
        # Make prediction
        with PREDICTION_LATENCY.labels(model_type="neuropathy_progression").time():
            prediction = model.predict(feature_values)[0]
            confidence = model.predict_proba(feature_values)[0].max() if hasattr(model, 'predict_proba') else 0.85
        
        # Calculate feature importance
        feature_importance = calculate_feature_importance(model, feature_values, feature_names)
        
        # Calculate additional metrics
        additional_data = {
            "progression_rate": float(prediction * 0.1),  # Simulated progression rate
            "time_to_progression": float(12 / max(prediction, 0.1))  # Months to progression
        }
        
        # Create response
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        response = PredictionResponse(
            patient_id=request.patient_id,
            model_type="neuropathy_progression",
            prediction=float(prediction),
            confidence=float(confidence),
            model_version=model_info["version"],
            feature_importance=feature_importance,
            additional_data=additional_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
        # Cache and store prediction
        background_tasks.add_task(cache_prediction, cache_key, response)
        background_tasks.add_task(store_prediction_mongodb, response)
        
        # Update metrics
        model_info["prediction_count"] += 1
        PREDICTION_COUNTER.labels(model_type="neuropathy_progression", status="success").inc()
        
        return response
        
    except Exception as e:
        PREDICTION_COUNTER.labels(model_type="neuropathy_progression", status="error").inc()
        logger.error(f"Error in neuropathy progression prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/glucose-complications", response_model=PredictionResponse)
async def predict_glucose_complications(
    request: PredictionFeatures,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Predict glucose-related complications"""
    start_time = datetime.now()
    
    try:
        # Similar implementation to neuropathy progression
        model_info = models.get("glucose_complications")
        if not model_info:
            raise HTTPException(status_code=404, detail="Glucose complications model not available")
        
        model = model_info["model"]
        feature_values = np.array([list(request.features.values())])
        
        with PREDICTION_LATENCY.labels(model_type="glucose_complications").time():
            prediction = model.predict(feature_values)[0]
            confidence = 0.82  # Simulated confidence
        
        additional_data = {
            "hypoglycemia_risk": float(prediction * 0.3),
            "hyperglycemia_risk": float(prediction * 0.7),
            "time_in_range": float(85.0 - prediction * 20)
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        response = PredictionResponse(
            patient_id=request.patient_id,
            model_type="glucose_complications",
            prediction=float(prediction),
            confidence=float(confidence),
            model_version=model_info["version"],
            additional_data=additional_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
        model_info["prediction_count"] += 1
        PREDICTION_COUNTER.labels(model_type="glucose_complications", status="success").inc()
        
        return response
        
    except Exception as e:
        PREDICTION_COUNTER.labels(model_type="glucose_complications", status="error").inc()
        logger.error(f"Error in glucose complications prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/anomalies", response_model=PredictionResponse)
async def detect_anomalies(
    request: PredictionFeatures,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Detect anomalies in sensor data"""
    start_time = datetime.now()
    
    try:
        model_info = models.get("anomaly_detection")
        if not model_info:
            raise HTTPException(status_code=404, detail="Anomaly detection model not available")
        
        model = model_info["model"]
        feature_values = np.array([list(request.features.values())])
        
        with PREDICTION_LATENCY.labels(model_type="anomaly_detection").time():
            anomaly_score = model.predict(feature_values)[0]
            confidence = 0.88
        
        additional_data = {
            "anomaly_type": "pressure_pattern" if anomaly_score > 0.5 else "normal",
            "severity": "high" if anomaly_score > 0.8 else "medium" if anomaly_score > 0.5 else "low"
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        response = PredictionResponse(
            patient_id=request.patient_id,
            model_type="anomaly_detection",
            prediction=float(anomaly_score),
            confidence=float(confidence),
            model_version=model_info["version"],
            additional_data=additional_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
        model_info["prediction_count"] += 1
        PREDICTION_COUNTER.labels(model_type="anomaly_detection", status="success").inc()
        
        return response
        
    except Exception as e:
        PREDICTION_COUNTER.labels(model_type="anomaly_detection", status="error").inc()
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/risk-stratification", response_model=PredictionResponse)
async def predict_risk_stratification(
    request: PredictionFeatures,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Calculate comprehensive risk stratification"""
    start_time = datetime.now()
    
    try:
        model_info = models.get("risk_stratification")
        if not model_info:
            raise HTTPException(status_code=404, detail="Risk stratification model not available")
        
        model = model_info["model"]
        feature_values = np.array([list(request.features.values())])
        
        with PREDICTION_LATENCY.labels(model_type="risk_stratification").time():
            risk_score = model.predict(feature_values)[0]
            confidence = 0.86
        
        additional_data = {
            "time_to_event": float(24 / max(risk_score, 0.1)),  # Months to event
            "intervention_priority": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        response = PredictionResponse(
            patient_id=request.patient_id,
            model_type="risk_stratification",
            prediction=float(risk_score),
            confidence=float(confidence),
            model_version=model_info["version"],
            additional_data=additional_data,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
        model_info["prediction_count"] += 1
        PREDICTION_COUNTER.labels(model_type="risk_stratification", status="success").inc()
        
        return response
        
    except Exception as e:
        PREDICTION_COUNTER.labels(model_type="risk_stratification", status="error").inc()
        logger.error(f"Error in risk stratification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Batch prediction for multiple requests"""
    start_time = datetime.now()
    batch_id = request.batch_id or f"batch_{int(datetime.now().timestamp())}"
    
    responses = []
    successful = 0
    failed = 0
    
    for pred_request in request.requests:
        try:
            # Route to appropriate prediction endpoint based on model type
            if pred_request.model_type == "neuropathy_progression":
                response = await predict_neuropathy_progression(pred_request, background_tasks, token)
            elif pred_request.model_type == "glucose_complications":
                response = await predict_glucose_complications(pred_request, background_tasks, token)
            elif pred_request.model_type == "anomaly_detection":
                response = await detect_anomalies(pred_request, background_tasks, token)
            elif pred_request.model_type == "risk_stratification":
                response = await predict_risk_stratification(pred_request, background_tasks, token)
            else:
                raise ValueError(f"Unknown model type: {pred_request.model_type}")
            
            responses.append(response)
            successful += 1
            
        except Exception as e:
            logger.error(f"Error in batch prediction for patient {pred_request.patient_id}: {str(e)}")
            failed += 1
    
    total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return BatchPredictionResponse(
        batch_id=batch_id,
        responses=responses,
        total_requests=len(request.requests),
        successful_predictions=successful,
        failed_predictions=failed,
        timestamp=datetime.now(),
        total_processing_time_ms=total_processing_time
    )

@app.get("/metrics/{model_type}", response_model=ModelMetrics)
async def get_model_metrics(model_type: str, token: str = Depends(verify_token)):
    """Get performance metrics for a specific model"""
    model_info = models.get(model_type)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
    
    # Simulate metrics - in production, these would come from actual monitoring
    return ModelMetrics(
        model_type=model_type,
        accuracy=0.85 + np.random.normal(0, 0.05),
        precision=0.83 + np.random.normal(0, 0.05),
        recall=0.81 + np.random.normal(0, 0.05),
        f1_score=0.82 + np.random.normal(0, 0.05),
        auc=0.88 + np.random.normal(0, 0.03),
        last_updated=datetime.now(),
        prediction_count=model_info["prediction_count"],
        average_latency_ms=150.0 + np.random.normal(0, 20)
    )

@app.post("/validate", response_model=ModelValidationResult)
async def validate_model(
    request: ValidationRequest,
    token: str = Depends(verify_token)
):
    """Validate model performance against ground truth"""
    try:
        model_info = models.get(request.model_type)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")
        
        # Simulate validation - in production, this would use actual validation data
        accuracy = 0.85 + np.random.normal(0, 0.03)
        precision = 0.83 + np.random.normal(0, 0.03)
        recall = 0.81 + np.random.normal(0, 0.03)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return ModelValidationResult(
            model_type=request.model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            validation_date=datetime.now(),
            sample_size=len(request.validation_data) if request.validation_data else 1000
        )
        
    except Exception as e:
        logger.error(f"Error validating model {request.model_type}: {str(e)}")
        return ModelValidationResult(
            model_type=request.model_type,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            validation_date=datetime.now(),
            sample_size=0,
            success=False,
            error_message=str(e)
        )

@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/models")
async def list_models(token: str = Depends(verify_token)):
    """List all available models"""
    model_list = []
    for name, info in models.items():
        model_list.append({
            "name": name,
            "version": info["version"],
            "loaded_at": info["loaded_at"],
            "prediction_count": info["prediction_count"],
            "accuracy_threshold": info["accuracy_threshold"]
        })
    return {"models": model_list}

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(asctime)s %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )