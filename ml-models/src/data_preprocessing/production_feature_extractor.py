import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

logger = logging.getLogger(__name__)

class ProductionFeatureExtractor:
    """Production-ready feature extractor for Smart Shoe ML models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_params = MinimalFCParameters()
        
    def extract_static_features(self, row: pd.Series) -> np.ndarray:
        """Extract static features from a single measurement."""
        try:
            features = []
            
            # Basic statistical features from current measurements
            for col in ['pinprick_threshold', 'temp_hot_threshold', 
                       'temp_cold_threshold', 'vibration_threshold']:
                if col in row.index:
                    value = float(row[col])
                    features.append(value)
                else:
                    features.append(0)
            
            # Add context features
            features.extend([
                float(row.get('test_completion_rate', 0)),
                float(row.get('medication_adherence', 0)),
                float(row.get('sleep_quality', 0)),
                float(row.get('stress_level', 0))
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting static features: {str(e)}")
            raise
            
    def extract_temporal_features(self, history: pd.DataFrame) -> np.ndarray:
        """Extract temporal features from patient history."""
        try:
            features = []
            
            # Sort by timestamp
            history = history.sort_values('timestamp')
            
            # Calculate temporal features for each measurement
            for col in ['pinprick_threshold', 'temp_hot_threshold', 
                       'temp_cold_threshold', 'vibration_threshold']:
                if col in history.columns:
                    values = history[col].astype(float)
                    if len(values) > 1:
                        # Rate of change
                        gradient = np.gradient(values)
                        features.extend([
                            gradient.mean(),
                            gradient.std(),
                            values.diff().mean(),
                            values.diff().std()
                        ])
                        
                        # Rolling statistics (last 2 weeks)
                        if len(values) >= 2:
                            rolling = values.rolling(window=2)
                            features.extend([
                                rolling.mean().mean(),
                                rolling.std().mean(),
                                rolling.max().mean(),
                                rolling.min().mean()
                            ])
                        else:
                            features.extend([0, 0, 0, 0])
                    else:
                        features.extend([0] * 8)  # 4 gradient + 4 rolling features
                else:
                    features.extend([0] * 8)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
            raise
            
    def extract_progression_features(self, history: pd.DataFrame) -> np.ndarray:
        """Extract progression features from patient history."""
        try:
            features = []
            
            # Sort by timestamp
            history = history.sort_values('timestamp')
            
            # Calculate progression features for each measurement
            for col in ['pinprick_threshold', 'temp_hot_threshold', 
                       'temp_cold_threshold', 'vibration_threshold']:
                if col in history.columns:
                    values = history[col].astype(float)
                    if len(values) > 1:
                        # Linear trend
                        x = np.arange(len(values))
                        A = np.vstack([x, np.ones(len(x))]).T
                        slope, _ = np.linalg.lstsq(A, values, rcond=None)[0]
                        
                        # Other progression metrics
                        total_change = values.iloc[-1] - values.iloc[0]
                        avg_change = total_change / (len(values) - 1)
                        
                        features.extend([
                            slope,
                            total_change,
                            avg_change,
                            np.std(np.diff(values))
                        ])
                    else:
                        features.extend([0, 0, 0, 0])
                else:
                    features.extend([0, 0, 0, 0])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting progression features: {str(e)}")
            raise
    
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract all features from the dataset."""
        try:
            # Convert timestamp to datetime if it's not already
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Initialize feature arrays
            all_features = []
            
            # Process each sample
            for _, row in data.iterrows():
                features = []
                
                # Add static features
                features.extend([
                    float(row['pinprick_threshold']),
                    float(row['temp_hot_threshold']),
                    float(row['temp_cold_threshold']),
                    float(row['vibration_threshold'])
                ])
                
                # Get patient history up to this point
                patient_id = row['patient_id']
                current_time = row['timestamp']
                history = data[
                    (data['patient_id'] == patient_id) & 
                    (data['timestamp'] <= current_time)
                ]
                
                # Add temporal features
                temporal = self.extract_temporal_features(history)
                features.extend(temporal)
                
                # Add progression features
                progression = self.extract_progression_features(history)
                features.extend(progression)
                
                all_features.append(features)
            
            # Convert to numpy array
            features_array = np.array(all_features)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features_array)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error extracting all features: {str(e)}")
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
        """Load a previously fitted scaler."""
        try:
            import joblib
            self.scaler = joblib.load(path)
            logger.info(f"Scaler loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise 