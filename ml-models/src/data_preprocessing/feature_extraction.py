import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from raw sensor data for ML models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_params = MinimalFCParameters()
        
    def extract_static_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract static features from sensor readings."""
        try:
            features = []
            
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Group by patient_id to get per-patient features
            for _, patient_data in data.groupby('patient_id'):
                patient_features = []
                
                # Basic statistical features
                for col in ['pinprick_threshold', 'temp_hot_threshold', 
                           'temp_cold_threshold', 'vibration_threshold']:
                    if col in patient_data.columns:
                        values = patient_data[col].astype(float)
                        if len(values) > 0:
                            patient_features.extend([
                                values.mean(),
                                values.std() if len(values) > 1 else 0,
                                values.min(),
                                values.max(),
                                values.quantile(0.25),
                                values.quantile(0.75)
                            ])
                        else:
                            patient_features.extend([0, 0, 0, 0, 0, 0])
                    else:
                        patient_features.extend([0, 0, 0, 0, 0, 0])
                
                features.append(patient_features)
            
            return np.array(features)
        except Exception as e:
            logger.error(f"Error extracting static features: {str(e)}")
            raise
            
    def extract_temporal_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract temporal features from time series data."""
        try:
            features = []
            window_sizes = [5, 10, 20]  # Different window sizes for temporal features
            
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Group by patient_id to get per-patient features
            for _, patient_data in data.groupby('patient_id'):
                # Ensure data is sorted by timestamp
                patient_data = patient_data.sort_values('timestamp')
                patient_features = []
                
                # Extract temporal features for each measurement
                for col in ['pinprick_threshold', 'temp_hot_threshold', 
                           'temp_cold_threshold', 'vibration_threshold']:
                    if col in patient_data.columns:
                        values = patient_data[col].astype(float)
                        
                        # Basic temporal features
                        if len(values) > 1:
                            try:
                                patient_features.extend([
                                    np.gradient(values).mean(),  # Average rate of change
                                    np.gradient(values).std(),   # Variability in rate of change
                                    np.diff(values).mean(),      # Average absolute change
                                    np.diff(values).std()        # Variability in absolute change
                                ])
                            except:
                                patient_features.extend([0, 0, 0, 0])
                        else:
                            patient_features.extend([0, 0, 0, 0])
                        
                        # Rolling window features
                        for window in window_sizes:
                            if len(values) >= window:
                                rolling = values.rolling(window=window)
                                try:
                                    patient_features.extend([
                                        rolling.mean().mean(),
                                        rolling.std().mean(),
                                        rolling.max().mean(),
                                        rolling.min().mean()
                                    ])
                                except:
                                    patient_features.extend([0, 0, 0, 0])
                            else:
                                patient_features.extend([0, 0, 0, 0])
                    else:
                        # Padding for missing columns
                        patient_features.extend([0] * (4 + len(window_sizes) * 4))
                
                features.append(patient_features)
            
            return np.array(features)
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
            raise
            
    def extract_progression_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract progression-related features."""
        try:
            features = []
            
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Group by patient_id to get per-patient features
            for _, patient_data in data.groupby('patient_id'):
                # Sort by timestamp
                patient_data = patient_data.sort_values('timestamp')
                patient_features = []
                
                # Calculate progression features for each measurement type
                for col in ['pinprick_threshold', 'temp_hot_threshold', 
                           'temp_cold_threshold', 'vibration_threshold']:
                    if col in patient_data.columns:
                        values = patient_data[col].astype(float)
                        
                        if len(values) > 1:
                            try:
                                # Linear trend (with numerical stability check)
                                x = np.arange(len(values))
                                A = np.vstack([x, np.ones(len(x))]).T
                                slope, _ = np.linalg.lstsq(A, values, rcond=None)[0]
                                
                                # Other progression features
                                total_change = values.iloc[-1] - values.iloc[0]
                                avg_change = total_change / len(values)
                                change_std = np.std(np.diff(values))
                                
                                patient_features.extend([
                                    slope,
                                    total_change,
                                    avg_change,
                                    change_std
                                ])
                            except:
                                patient_features.extend([0, 0, 0, 0])
                        else:
                            patient_features.extend([0, 0, 0, 0])
                    else:
                        patient_features.extend([0, 0, 0, 0])
                
                features.append(patient_features)
            
            return np.array(features)
        except Exception as e:
            logger.error(f"Error extracting progression features: {str(e)}")
            raise
            
    def extract_all_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract all features for model training."""
        try:
            features = {
                'static': self.extract_static_features(data),
                'temporal': self.extract_temporal_features(data),
                'progression': self.extract_progression_features(data)
            }
            
            # Scale features
            for feature_type in features:
                if len(features[feature_type].shape) == 1:
                    features[feature_type] = features[feature_type].reshape(-1, 1)
                features[feature_type] = self.scaler.fit_transform(features[feature_type])
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting all features: {str(e)}")
            raise
            
    def prepare_training_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare features for model training with metadata."""
        try:
            features = self.extract_all_features(data)
            
            # Combine all features
            combined_features = np.concatenate([
                features['static'].flatten(),
                features['temporal'].flatten(),
                features['progression'].flatten()
            ])
            
            # Create feature metadata
            feature_metadata = {
                'static_features': list(range(len(features['static'].flatten()))),
                'temporal_features': list(range(
                    len(features['static'].flatten()),
                    len(features['static'].flatten()) + len(features['temporal'].flatten())
                )),
                'progression_features': list(range(
                    len(features['static'].flatten()) + len(features['temporal'].flatten()),
                    len(combined_features)
                ))
            }
            
            return combined_features, feature_metadata
            
        except Exception as e:
            logger.error(f"Error preparing training features: {str(e)}")
            raise
            
    def save_scaler(self, path: str) -> None:
        """Save the fitted scaler."""
        try:
            import joblib
            joblib.dump(self.scaler, path)
            logger.info(f"Scaler saved to {path}")
        except Exception as e:
            logger.error(f"Error saving scaler: {str(e)}")
            raise
            
    def load_scaler(self, path: str) -> None:
        """Load a fitted scaler."""
        try:
            import joblib
            self.scaler = joblib.load(path)
            logger.info(f"Scaler loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise 