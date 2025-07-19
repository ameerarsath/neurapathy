import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class ProgressionTracker:
    """
    Model for tracking neuropathy progression over time.
    Uses time-series analysis and regression to detect changes.
    """
    
    def __init__(self, lookback_window=3, prediction_horizon=1):
        self.lookback_window = lookback_window  # weeks
        self.prediction_horizon = prediction_horizon  # weeks
        self.models = {}  # One model per test type
        self.scalers = {}
        self.progression_thresholds = {
            'pinprick': 0.15,  # 15% change
            'temperature': 0.20,  # 20% change
            'vibration': 0.25   # 25% change
        }
        self.model = GradientBoostingRegressor(random_state=42)
        self.is_fitted = False

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            model_state = {
                'model': self.model,
                'is_fitted': self.is_fitted
            }
            joblib.dump(model_state, filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"No model file found at {filepath}")
            
            model_state = joblib.load(filepath)
            self.model = model_state['model']
            self.is_fitted = model_state['is_fitted']
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def prepare_features(self, data: pd.DataFrame, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time-series features for progression tracking."""
        # Sort by weeks_monitored
        data_sorted = data.sort_values('weeks_monitored')
        
        features = []
        targets = []
        
        test_types = [
            'pinprick_threshold_avg', 
            'temp_hot_threshold_avg', 
            'temp_cold_threshold_avg', 
            'vibration_threshold_avg'
        ]
        
        for i in range(self.lookback_window, len(data_sorted) - self.prediction_horizon):
            # Historical window
            window_data = data_sorted.iloc[i-self.lookback_window:i]
            
            # Extract features from window
            feature_vector = []
            for test_type in test_types:
                values = window_data[test_type].values
                feature_vector.extend([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                    values[-1] - values[0],  # trend
                    np.percentile(values, 75) - np.percentile(values, 25)  # IQR
                ])
            
            # Add additional features
            feature_vector.extend([
                window_data['weeks_monitored'].mean(),
                window_data['response_time_avg'].mean(),
                window_data['symptom_score_total'].mean()
            ])
            
            features.append(feature_vector)
            
            # Target: future values
            future_data = data_sorted.iloc[i:i+self.prediction_horizon]
            target_vector = [future_data[test_type].mean() for test_type in test_types]
            targets.append(target_vector)
        
        return np.array(features), np.array(targets)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **params) -> None:
        """Train progression tracking model."""
        try:
            # Update model parameters if provided
            if params:
                self.model.set_params(**params)
            
            # Ensure y is 2D array
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            
            # Fit model
            self.model.fit(X, y)
            self.is_fitted = True
            
            return self
            
        except Exception as e:
            logger.error(f"Error training progression tracker: {str(e)}")
            raise
    
    def predict_progression(self, X: np.ndarray) -> np.ndarray:
        """Predict progression rate."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            predictions = self.model.predict(X)
            # Ensure output is 2D array
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            y_pred = self.predict_progression(X)
            
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def _extract_single_feature_vector(self, window_data: pd.DataFrame, test_type: str) -> List[float]:
        """Extract feature vector for a single prediction."""
        threshold_col = f"{test_type}_threshold"
        
        if threshold_col not in window_data.columns:
            return [0] * 25  # Default feature vector
        
        values = window_data[threshold_col].values
        
        # Basic statistical features
        features = [
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            values[-1] - values[0],  # trend
            np.percentile(values, 75) - np.percentile(values, 25)  # IQR
        ]
        
        # Add features for other test types
        other_types = ['pinprick_threshold', 'temp_hot_threshold', 'temp_cold_threshold', 'vibration_threshold']
        for other_type in other_types:
            if other_type in window_data.columns and other_type != threshold_col:
                other_values = window_data[other_type].values
                features.extend([
                    np.mean(other_values),
                    np.std(other_values),
                    np.min(other_values),
                    np.max(other_values),
                    other_values[-1] - other_values[0],
                    np.percentile(other_values, 75) - np.percentile(other_values, 25)
                ])
        
        # Pad to expected length
        while len(features) < 25:
            features.append(0)
        
        return features[:25]
    
    def _classify_progression(self, progression_rate: float, test_type: str) -> str:
        """Classify progression severity."""
        threshold = self.progression_thresholds.get(test_type.split('_')[0], 0.2)
        
        if abs(progression_rate) < threshold * 0.5:
            return 'stable'
        elif abs(progression_rate) < threshold:
            return 'mild_change'
        elif abs(progression_rate) < threshold * 2:
            return 'moderate_change'
        else:
            return 'severe_change'