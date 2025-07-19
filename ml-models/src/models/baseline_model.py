import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import logging
from typing import Dict, List, Tuple, Optional

class BaselineModel:
    """
    Model for establishing patient-specific baselines for neuropathy testing.
    Uses clustering and statistical methods to determine normal response patterns.
    """
    
    def __init__(self, contamination=0.1):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.kmeans = KMeans(n_clusters=3, random_state=42)  # Normal, Mild, Severe sensitivity
        self.baseline_stats = {}
        self.is_fitted = False
        
    def fit(self, sensor_data: pd.DataFrame, patient_id: str) -> None:
        """
        Establish baseline for a specific patient.
        
        Args:
            sensor_data: DataFrame with columns ['pinprick_threshold_avg', 'temp_hot_threshold_avg', 
                        'temp_cold_threshold_avg', 'vibration_threshold_avg', 'response_time_avg']
            patient_id: Unique identifier for the patient
        """
        try:
            # Feature engineering
            features = self._extract_features(sensor_data)
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect outliers in initial readings
            outlier_mask = self.isolation_forest.fit_predict(features_scaled) == 1
            clean_data = features_scaled[outlier_mask]
            
            # Check if we have enough data for clustering
            if len(clean_data) >= 3:
                # Cluster sensitivity levels
                self.kmeans.fit(clean_data)
                cluster_centers = self.kmeans.cluster_centers_
            else:
                # Not enough data for clustering, use simple thresholds
                logging.warning(f"Insufficient data for clustering for patient {patient_id}. Using simple thresholds.")
                cluster_centers = np.array([
                    np.mean(features_scaled, axis=0),  # normal
                    np.mean(features_scaled, axis=0) + np.std(features_scaled, axis=0),  # mild loss
                    np.mean(features_scaled, axis=0) + 2 * np.std(features_scaled, axis=0)  # severe loss
                ])
            
            # Calculate statistical baselines
            self.baseline_stats[patient_id] = {
                'pinprick_mean': np.mean(sensor_data['pinprick_threshold_avg']),
                'pinprick_std': np.std(sensor_data['pinprick_threshold_avg']) or 1.0,  # Fallback if std=0
                'temp_hot_mean': np.mean(sensor_data['temp_hot_threshold_avg']),
                'temp_hot_std': np.std(sensor_data['temp_hot_threshold_avg']) or 1.0,
                'temp_cold_mean': np.mean(sensor_data['temp_cold_threshold_avg']),
                'temp_cold_std': np.std(sensor_data['temp_cold_threshold_avg']) or 1.0,
                'vibration_mean': np.mean(sensor_data['vibration_threshold_avg']),
                'vibration_std': np.std(sensor_data['vibration_threshold_avg']) or 1.0,
                'cluster_centers': cluster_centers,
                'total_readings': len(sensor_data)
            }
            
            self.is_fitted = True
            logging.info(f"Baseline established for patient {patient_id}")
            
        except Exception as e:
            logging.error(f"Error fitting baseline model: {str(e)}")
            raise
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract relevant features for baseline establishment."""
        features = []
        
        # Basic threshold features
        features.extend([
            data['pinprick_threshold_avg'].values,
            data['temp_hot_threshold_avg'].values,
            data['temp_cold_threshold_avg'].values,
            data['vibration_threshold_avg'].values
        ])
        
        # Time-based features (if multiple readings)
        if len(data) > 1:
            features.extend([
                data['pinprick_threshold_avg'].rolling(window=3).mean().fillna(data['pinprick_threshold_avg']).values,
                data['vibration_threshold_avg'].rolling(window=3).std().fillna(0).values
            ])
        
        return np.column_stack(features)

    # ...existing code for predict_sensitivity_level, save_model, and load_model...
    
    def predict_sensitivity_level(self, new_data: pd.DataFrame, patient_id: str) -> np.ndarray:
        """Predict sensitivity level compared to baseline."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        features = self._extract_features(new_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict cluster membership
        clusters = self.kmeans.predict(features_scaled)
        
        # Map clusters to sensitivity levels
        sensitivity_map = {0: 'normal', 1: 'mild_loss', 2: 'severe_loss'}
        return np.array([sensitivity_map[c] for c in clusters])
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest,
            'kmeans': self.kmeans,
            'baseline_stats': self.baseline_stats,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.isolation_forest = model_data['isolation_forest']
        self.kmeans = model_data['kmeans']
        self.baseline_stats = model_data['baseline_stats']
        self.is_fitted = model_data['is_fitted']
