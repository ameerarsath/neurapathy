"""
Secure model loading and validation.
"""
import os
import logging
from typing import Any, Optional
import joblib
from .security_config import SecurityConfig, ModelMetadata
from .model_security import ModelSecurity

class SecureModelLoader:
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize secure model loader."""
        self.config = config or SecurityConfig()
        self.security = ModelSecurity()
        
    def load_model(self, model_path: str, expected_type: type) -> Any:
        """Load model with security checks."""
        self._validate_path(model_path)
        
        if self.config.checksum_verification:
            checksums_path = os.path.join(
                os.path.dirname(model_path), 'model_checksums.json'
            )
            if not self.security.verify_model_integrity(model_path, checksums_path):
                raise ValueError(f"Model integrity check failed: {model_path}")
        
        model = joblib.load(model_path)
        if not isinstance(model, expected_type):
            raise TypeError(f"Model is not of expected type {expected_type.__name__}")
            
        return model
        
    def _validate_path(self, path: str) -> None:
        """Validate model file path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        if not any(path.endswith(ext) for ext in self.config.allowed_model_formats):
            raise ValueError(
                f"Invalid model format. Allowed: {self.config.allowed_model_formats}"
            )
            
    def save_model(self, model: Any, path: str, metadata: ModelMetadata) -> None:
        """Save model with security measures."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Encrypt and save model if enabled
        if self.config.encryption_enabled:
            model_bytes = joblib.dumps(model)
            encrypted = self.security.encrypt_data(model_bytes)
            with open(path, 'wb') as f:
                f.write(encrypted)
        else:
            joblib.dump(model, path)
            
        # Update checksums
        self._update_checksums(path, metadata)
