from pydantic import BaseSettings
from typing import Dict, Any
import os

class ProductionConfig(BaseSettings):
    # API Settings
    API_WORKERS: int = 4
    API_TIMEOUT: int = 30
    MAX_REQUESTS_PER_MINUTE: int = 1000
    
    # Model Settings
    MODEL_REFRESH_INTERVAL: int = 3600  # 1 hour
    CONFIDENCE_THRESHOLD: float = 0.85
    BATCH_SIZE: int = 32
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 60
    
    # Performance
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 300  # 5 minutes
    
    # Security
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_REQUEST_LOGGING: bool = True
    
    # Model Monitoring
    DRIFT_DETECTION_WINDOW: int = 1000  # samples
    DRIFT_THRESHOLD: float = 0.1
    
    # Error Handling
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1
    
    # Resource Limits
    MAX_MEMORY_MB: int = 4096
    CPU_LIMIT: float = 0.8  # 80% CPU usage threshold
    
    class Config:
        env_file = ".env"

    def get_model_serving_config(self) -> Dict[str, Any]:
        return {
            "batch_size": self.BATCH_SIZE,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "max_memory_mb": self.MAX_MEMORY_MB,
            "cpu_limit": self.CPU_LIMIT
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        return {
            "metrics_enabled": self.ENABLE_METRICS,
            "metrics_port": self.METRICS_PORT,
            "health_check_interval": self.HEALTH_CHECK_INTERVAL,
            "drift_detection_window": self.DRIFT_DETECTION_WINDOW,
            "drift_threshold": self.DRIFT_THRESHOLD
        }

# Initialize configuration
config = ProductionConfig() 