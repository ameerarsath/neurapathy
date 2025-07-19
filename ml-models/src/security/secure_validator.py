"""
Secure model validation with enhanced security features.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .security_config import SecurityConfig
from .model_registry import ModelRegistry
from .enhanced_security import ModelSecurity

class SecureModelValidator:
    """Secure model validation with enhanced security features."""
    
    def __init__(self, 
                 models_dir: str,
                 security_config: Optional[SecurityConfig] = None):
        """Initialize secure model validator."""
        self.models_dir = models_dir
        self.config = security_config or SecurityConfig()
        
        # Initialize security components
        registry_dir = os.path.join(models_dir, 'registry')
        self.registry = ModelRegistry(registry_dir, self.config)
        self.security = self.registry.security
        self.audit = self.registry.audit
        
        # Store validation results
        self.validation_results = {}
        
    def validate_model(self,
                      model_id: str,
                      test_data: pd.DataFrame,
                      expected_outputs: Optional[pd.DataFrame] = None) -> Dict:
        """Validate a model with security measures."""
        try:
            # Verify model integrity
            model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
            if not self.registry.verify_model(model_path, model_id):
                raise ValueError(f"Model integrity check failed: {model_id}")
                
            # Validate input data
            if not self._validate_input_data(test_data):
                raise ValueError("Invalid input data format")
                
            # Load model metadata
            metadata = self.registry.get_model_metadata(model_id)
            if not metadata:
                raise ValueError(f"No metadata found for model: {model_id}")
                
            # Run validation
            results = self._run_validation(
                model_path,
                test_data,
                expected_outputs
            )
            
            # Store and audit results
            self.validation_results[model_id] = results
            self.audit.log_validation_event(
                model_type=metadata.model_id.split('_')[0],
                validation_type='FULL',
                status='SUCCESS',
                details=results
            )
            
            return results
            
        except Exception as e:
            error_details = {'error': str(e), 'model_id': model_id}
            self.audit.log_validation_event(
                model_type='unknown',
                validation_type='FULL',
                status='FAILED',
                details=error_details
            )
            return error_details
            
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format and content."""
        try:
            # Check data size
            if len(data) > self.config.max_batch_size:
                return False
                
            # Check for required columns
            required_cols = ['patient_id', 'timestamp']
            if not all(col in data.columns for col in required_cols):
                return False
                
            # Check for invalid values
            if not data.select_dtypes(include=[np.number]).apply(np.isfinite).all().all():
                return False
                
            return True
            
        except Exception:
            return False
            
    def _run_validation(self,
                       model_path: str,
                       test_data: pd.DataFrame,
                       expected_outputs: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Run model validation with security measures."""
        import joblib
        
        # Load model securely
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            if self.config.encryption_enabled:
                model_bytes = self.security.decrypt_data(model_bytes)
            model = joblib.load(model_bytes)
            
        # Make predictions
        predictions = model.predict(test_data)
        
        # Calculate metrics
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_samples': len(test_data),
            'data_coverage': self._calculate_coverage(test_data)
        }
        
        if expected_outputs is not None:
            metrics.update(self._calculate_performance_metrics(
                predictions, expected_outputs
            ))
            
        return metrics
        
    def _calculate_coverage(self, data: pd.DataFrame) -> float:
        """Calculate data coverage percentage."""
        return 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        
    def _calculate_performance_metrics(self,
                                    predictions: np.ndarray,
                                    actuals: pd.DataFrame) -> Dict[str, float]:
        """Calculate model performance metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        metrics = {}
        try:
            metrics['mae'] = float(mean_absolute_error(actuals, predictions))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(actuals, predictions)))
        except Exception as e:
            self.audit.log_event(
                event_type='METRICS',
                action='CALCULATE',
                resource='performance_metrics',
                status='FAILED',
                details={'error': str(e)}
            )
            
        return metrics
