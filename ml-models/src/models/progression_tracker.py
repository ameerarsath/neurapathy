import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')
import joblib 
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
    # ...existing code...

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'lookback_window': self.lookback_window,
                'prediction_horizon': self.prediction_horizon,
                'progression_thresholds': self.progression_thresholds
            }
            joblib.dump(model_data, filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.lookback_window = model_data['lookback_window']
            self.prediction_horizon = model_data['prediction_horizon']
            self.progression_thresholds = model_data['progression_thresholds']
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

# ...existing code...    
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
    
    def fit(self, data: pd.DataFrame, patient_id: str) -> None:
        """Train progression tracking model for a patient."""
        try:
            # Prepare features
            X, y = self.prepare_features(data, patient_id)
            
            if len(X) < 2:  # Need at least 2 readings for progression
                logging.warning(f"Insufficient data for patient {patient_id}")
                return
            
            # Initialize models for each test type
            test_types = ['pinprick', 'temp_hot', 'temp_cold', 'vibration']
            
            for i, test_type in enumerate(test_types):
                if i < y.shape[1]:
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train model
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                    model.fit(X_scaled, y[:, i])
                    
                    # Store model and scaler
                    self.models[f"{patient_id}_{test_type}"] = model
                    self.scalers[f"{patient_id}_{test_type}"] = scaler
            
            logging.info(f"Progression tracker trained for patient {patient_id}")
            
        except Exception as e:
            logging.error(f"Error training progression tracker: {str(e)}")
            raise
    
    def predict_progression(self, recent_data: pd.DataFrame, patient_id: str) -> Dict:
        """Predict future progression for a patient."""
        predictions = {}
        
        try:
            # Prepare features from recent data
            if len(recent_data) < self.lookback_window:
                return {"error": "Insufficient recent data for prediction"}
            
            # Get last window of data
            window_data = recent_data.tail(self.lookback_window)
            
            test_types = ['pinprick', 'temp_hot', 'temp_cold', 'vibration']
            
            for test_type in test_types:
                model_key = f"{patient_id}_{test_type}"
                
                if model_key in self.models:
                    # Extract features (simplified version)
                    feature_vector = self._extract_single_feature_vector(window_data, test_type)
                    
                    # Scale features
                    feature_scaled = self.scalers[model_key].transform([feature_vector])
                    
                    # Predict
                    prediction = self.models[model_key].predict(feature_scaled)[0]
                    
                    # Calculate progression rate
                    current_value = window_data[f"{test_type}_threshold"].iloc[-1]
                    progression_rate = (prediction - current_value) / current_value
                    
                    predictions[test_type] = {
                        'predicted_value': prediction,
                        'current_value': current_value,
                        'progression_rate': progression_rate,
                        'severity': self._classify_progression(progression_rate, test_type)
                    }
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error predicting progression: {str(e)}")
            return {"error": str(e)}
    
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