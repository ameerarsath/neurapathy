"""
Model security configuration and base classes.
"""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SecurityConfig:
    """Security configuration for model validation."""
    encryption_enabled: bool = True
    data_validation_required: bool = True
    access_logging_enabled: bool = True
    max_batch_size: int = 1000
    allowed_model_formats: List[str] = None
    checksum_verification: bool = True
    
    def __post_init__(self):
        if self.allowed_model_formats is None:
            self.allowed_model_formats = ['.pkl', '.h5', '.joblib']

@dataclass
class ModelMetadata:
    """Metadata for model versioning and tracking."""
    model_id: str
    version: str
    created_at: str
    checksum: str
    algorithm: str = 'sha256'
