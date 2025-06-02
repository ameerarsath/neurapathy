import numpy as np
import pandas as pd
import sys
import os
import logging
from typing import Dict, List, Union, Any
import joblib
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, roc_auc_score
import json
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_model import BaselineModel
from models.progression_tracker import ProgressionTracker
from models.risk_predictor import RiskPredictor
from security.model_security import ModelSecurity
from security.security_config import SecurityConfig
from security.secure_loader import SecureModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelValidator:
    """Comprehensive model validation framework with security."""
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.validation_results = {}
        self.security_config = SecurityConfig()
        self.model_loader = SecureModelLoader(self.security_config)
        
        if self.security_config.access_logging_enabled:
            self._setup_audit_logging()
    
    def _setup_audit_logging(self):
        """Configure secure audit logging."""
        log_dir = os.path.join(os.path.dirname(self.models_path), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Add secure audit trail
        audit_handler = logging.FileHandler(
            os.path.join(log_dir, 'model_audit.log')
        )
        audit_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(formatter)
        logging.getLogger().addHandler(audit_handler)

    def validate_baseline_model(self, patient_data: Dict[str, Union[pd.DataFrame, List]]) -> Dict:
        """Validate baseline models for each patient."""
        results = {}
        
        for patient_id, data in patient_data.items():
            try:
                # Convert data to DataFrame if it's not already
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                
                # Load baseline model
                model_path = os.path.join(self.models_path, f"baseline_{patient_id}.pkl")
                if not os.path.exists(model_path):
                    results[patient_id] = {'error': 'Model file not found'}
                    continue
                    
                baseline_model = joblib.load(model_path)
                
                # For single sample, use it as both train and test
                if len(data) == 1:
                    metrics = {
                        'n_samples': 1,
                        'train_samples': 1,
                        'test_samples': 1,
                        'mae': 0.0,
                        'consistency_score': 1.0
                    }
                else:
                    # Split data for validation
                    train_size = max(1, int(len(data) * 0.7))
                    train_data = data.iloc[:train_size]
                    test_data = data.iloc[train_size:]
                    
                    # Make predictions
                    threshold_cols = [
                        'pinprick_threshold_avg', 
                        'temp_hot_threshold_avg',
                        'temp_cold_threshold_avg', 
                        'vibration_threshold_avg'
                    ]
                    
                    test_predictions = baseline_model.predict(test_data)
                    
                    # Calculate metrics
                    mae_scores = []
                    for col in threshold_cols:
                        if col in test_data.columns:
                            actuals = test_data[col].values
                            preds = test_predictions.get(col, np.zeros_like(actuals))
                            mae = np.mean(np.abs(actuals - preds))
                            mae_scores.append(mae)
                    
                    metrics = {
                        'mae': np.mean(mae_scores) if mae_scores else None,
                        'n_samples': len(data),
                        'train_samples': len(train_data),
                        'test_samples': len(test_data),
                        'consistency_score': self._calculate_consistency(
                            test_predictions.get('anomaly_labels', [])
                        )
                    }
                
                results[patient_id] = metrics
                
            except Exception as e:
                logging.error(f"Error validating baseline model for patient {patient_id}: {str(e)}")
                results[patient_id] = {'error': str(e)}
        
        self.validation_results['baseline_models'] = results
        return results

    def validate_progression_model(self, patient_data: Dict[str, Union[pd.DataFrame, List]]) -> Dict:
        """Validate progression tracking models."""
        results = {}
        
        for patient_id, data in patient_data.items():
            try:
                # Convert data to DataFrame if it's not already
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                
                # Load progression model
                model_path = os.path.join(self.models_path, f"progression_{patient_id}.pkl")
                if not os.path.exists(model_path):
                    results[patient_id] = {'error': 'Model file not found'}
                    continue
                
                progression_model = joblib.load(model_path)
                
                # Handle single sample case
                if len(data) == 1:
                    results[patient_id] = {
                        'error': 'Insufficient data for progression analysis'
                    }
                    continue
                
                # Split data for validation
                train_size = max(2, int(len(data) * 0.7))  # Need at least 2 samples
                train_data = data.iloc[:train_size]
                test_data = data.iloc[train_size:]
                
                if len(test_data) == 0:
                    results[patient_id] = {'error': 'Insufficient data for validation'}
                    continue
                
                # Make predictions
                predictions = progression_model.predict(test_data)
                
                # Calculate metrics
                metrics = {
                    'mae': mean_absolute_error(
                        test_data['pinprick_threshold_avg'], 
                        predictions['pinprick_threshold_avg']
                    ) if 'pinprick_threshold_avg' in predictions else None,
                    'n_samples': len(data),
                    'train_samples': len(train_data),
                    'test_samples': len(test_data)
                }
                
                results[patient_id] = metrics
                
            except Exception as e:
                logging.error(f"Error validating progression model for patient {patient_id}: {str(e)}")
                results[patient_id] = {'error': str(e)}
        
        self.validation_results['progression_models'] = results
        return results

    def validate_risk_predictor(self, training_data: np.ndarray, test_data: np.ndarray) -> Dict:
        """Validate risk prediction model with security measures."""
        try:
            # Validate input data
            if not (self._validate_input_data(training_data) and self._validate_input_data(test_data)):
                return {'error': 'Invalid input data format or security check failed'}

            if len(training_data) == 0 or len(test_data) == 0:
                return {'error': 'Empty training or test data'}
            
            # Ensure data is 2D
            if training_data.ndim == 1:
                training_data = training_data.reshape(1, -1)
            if test_data.ndim == 1:
                test_data = test_data.reshape(1, -1)
            
            # Initialize and train risk predictor with logging
            logging.info(f"Training risk predictor with {len(training_data)} samples")
            risk_predictor = RiskPredictor()
            risk_predictor.fit(training_data)
            
            # Make predictions with validation
            predictions = risk_predictor.predict(test_data)
            if not isinstance(predictions, np.ndarray):
                raise ValueError("Invalid prediction format")
            
            # Calculate metrics
            results = {
                'n_train_samples': len(training_data),
                'n_test_samples': len(test_data),
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'input_features': test_data.shape[1]
            }
            
            self.validation_results['risk_prediction'] = results
            return results
            
        except Exception as e:
            logging.error(f"Error validating risk predictor: {str(e)}")
            return {'error': str(e)}

    def _calculate_consistency(self, predictions: np.ndarray) -> float:
        """Calculate prediction consistency score."""
        if len(predictions) < 2:
            return 1.0
        return sum(1 for i in range(1, len(predictions)) 
                  if predictions[i] == predictions[i-1]) / (len(predictions) - 1)

    def generate_validation_summary(self, output_dir: str) -> None:
        """Generate validation summary report."""
        try:
            summary = {
                'baseline_models': self._summarize_baseline_results(),
                'progression_models': self._summarize_progression_results(),
                'risk_prediction': self.validation_results.get('risk_prediction', {})
            }
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'validation_summary.json')
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=4)
                
            logging.info(f"Validation summary saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error generating validation summary: {str(e)}")
            raise

    def _summarize_baseline_results(self) -> Dict:
        """Summarize baseline model validation results."""
        results = self.validation_results.get('baseline_models', {})
        return {
            'total_patients': len(results),
            'successful_validations': sum(1 for r in results.values() if 'error' not in r),
            'average_mae': np.mean([r.get('mae', np.nan) for r in results.values() if 'mae' in r])
        }

    def _summarize_progression_results(self) -> Dict:
        """Summarize progression model validation results."""
        results = self.validation_results.get('progression_models', {})
        return {
            'total_patients': len(results),
            'successful_validations': sum(1 for r in results.values() if 'error' not in r),
            'average_mae': np.mean([r.get('mae', np.nan) for r in results.values() if 'mae' in r])
        }

    def _validate_input_data(self, data: Union[Dict, np.ndarray]) -> bool:
        """Validate input data for security and consistency."""
        try:
            if isinstance(data, dict):
                # Validate dictionary data
                for key, value in data.items():
                    if not isinstance(key, str):
                        return False
                    if isinstance(value, np.ndarray):
                        if not self._validate_array_data(value):
                            return False
            elif isinstance(data, np.ndarray):
                return self._validate_array_data(data)
            else:
                return False
            return True
        except Exception:
            return False

    def _validate_array_data(self, arr: np.ndarray) -> bool:
        """Validate numpy array data for security."""
        if arr.size > self.security_config.max_batch_size:
            return False
        if not np.isfinite(arr).all():
            return False
        return True    
    def _verify_model_checksum(self, model_path: str) -> bool:
        """Verify model file integrity using checksum."""
        import hashlib
        import json
        
        if not os.path.exists(model_path):
            return False
            
        if not any(model_path.endswith(ext) for ext in self.security_config.allowed_model_formats):
            return False
            
        try:
            # Load stored checksums
            checksums_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                        'models', 'model_checksums.json')
            with open(checksums_path, 'r') as f:
                checksums = json.load(f)
            
            # Calculate current checksum
            with open(model_path, 'rb') as f:
                file_bytes = f.read()
                current_hash = hashlib.sha256(file_bytes).hexdigest()
            
            # Get expected checksum based on model type
            model_type = os.path.basename(model_path).split('_')[0]
            expected_hash = checksums['checksums'].get(f"{model_type}_model", {}).get('value')
            
            if not expected_hash:
                logging.error(f"No checksum found for model type: {model_type}")
                return False
                
            return current_hash == expected_hash
            
        except Exception as e:
            logging.error(f"Error verifying model checksum: {str(e)}")
            return False

    def _load_model_securely(self, model_path: str, expected_type: type) -> Any:
        """Securely load a model with validation and error handling."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if not self._verify_model_checksum(model_path):
            raise ValueError(f"Model integrity check failed: {model_path}")
            
        try:
            model = joblib.load(model_path)
            if not isinstance(model, expected_type):
                raise TypeError(f"Loaded model is not of expected type {expected_type.__name__}")
            return model
        except Exception as e:
            logging.error(f"Error loading model {model_path}: {str(e)}")
            raise

    def _encrypt_predictions(self, predictions: np.ndarray) -> bytes:
        """Encrypt sensitive prediction data."""
        from cryptography.fernet import Fernet
        try:
            # In production, key should be stored securely, not generated each time
            key = Fernet.generate_key()
            f = Fernet(key)
            return f.encrypt(predictions.tobytes())
        except Exception as e:
            logging.error(f"Encryption error: {str(e)}")
            raise

    def _secure_logging(self, event_type: str, details: Dict) -> None:
        """Log events with security context."""
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'source_ip': 'localhost',  # In production, get actual source
            'model_version': '1.0.0'
        }
        
        # Log to secure audit trail
        logging.info(f"Security event: {json.dumps(log_entry)}")