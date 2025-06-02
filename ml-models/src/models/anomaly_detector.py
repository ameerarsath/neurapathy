import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
import logging
from typing import Dict, List, Tuple

class AnomalyDetector:
    """
    Detects anomalous readings that might indicate sensor malfunction,
    unusual patient behavior, or rapid neuropathy progression.
    """
    
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.elliptic_envelope = EllipticEnvelope(contamination=contamination)
        self.scaler = RobustScaler()
        self.patient_models = {}
        self.anomaly_history = {}
        
    def fit_patient_profile(self, data: pd.DataFrame, patient_id: str) -> None:
        """Fit anomaly detection model for a specific patient."""
        try:
            # Prepare features
            features = self._prepare_anomaly_features(data)
            
            if len(features) < 10:
                logging.warning(f"Insufficient data for anomaly detection - patient {patient_id}")
                return
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit anomaly detectors
            isolation_model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            isolation_model.fit(features_scaled)
            
            elliptic_model = EllipticEnvelope(contamination=self.contamination)
            elliptic_model.fit(features_scaled)
            
            # Store models
            self.patient_models[patient_id] = {
                'isolation_forest': isolation_model,
                'elliptic_envelope': elliptic_model,
                'scaler': self.scaler,
                'baseline_stats': self._calculate_baseline_stats(data)
            }
            
            logging.info(f"Anomaly detector fitted for patient {patient_id}")
            
        except Exception as e:
            logging.error(f"Error fitting anomaly detector: {str(e)}")
            raise
    
    def detect_anomalies(self, new_data: pd.DataFrame, patient_id: str) -> Dict:
        """Detect anomalies in new readings."""
        if patient_id not in self.patient_models:
            return {"error": "Patient model not found"}
        
        try:
            # Prepare features
            features = self._prepare_anomaly_features(new_data)
            
            if len(features) == 0:
                return {"error": "No valid features extracted"}
            
            # Get patient model
            model_data = self.patient_models[patient_id]
            
            # Scale features
            features_scaled = model_data['scaler'].transform(features)
            
            # Detect anomalies using both methods
            isolation_anomalies = model_data['isolation_forest'].predict(features_scaled)
            elliptic_anomalies = model_data['elliptic_envelope'].predict(features_scaled)
            
            # Combine results (anomaly if either method detects it)
            combined_anomalies = (isolation_anomalies == -1) | (elliptic_anomalies == -1)
            
            # Calculate anomaly scores
            isolation_scores = model_data['isolation_forest'].decision_function(features_scaled)
            
            # Analyze anomaly types
            anomaly_analysis = self._analyze_anomaly_types(
                new_data, combined_anomalies, model_data['baseline_stats']
            )
            
            # Store in history
            if patient_id not in self.anomaly_history:
                self.anomaly_history[patient_id] = []
            
            self.anomaly_history[patient_id].append({
                'timestamp': pd.Timestamp.now(),
                'anomalies_detected': np.sum(combined_anomalies),
                'total_readings': len(combined_anomalies),
                'anomaly_types': anomaly_analysis
            })
            
            return {
                'anomalies_detected': combined_anomalies.tolist(),
                'anomaly_scores': isolation_scores.tolist(),
                'anomaly_rate': np.sum(combined_anomalies) / len(combined_anomalies),
                'anomaly_analysis': anomaly_analysis,
                'recommendations': self._generate_recommendations(anomaly_analysis)
            }
            
        except Exception as e:
            logging.error(f"Error detecting anomalies: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_anomaly_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection."""
        features = []
        
        # Basic threshold features
        threshold_cols = ['pinprick_threshold', 'temp_hot_threshold', 
                         'temp_cold_threshold', 'vibration_threshold']
        
        for col in threshold_cols:
            if col in data.columns:
                values = data[col].dropna().values
                if len(values) > 0:
                    features.append(values)
        
        if not features:
            return np.array([])
        
        # Ensure all arrays have the same length
        min_length = min(len(arr) for arr in features)
        features = [arr[:min_length] for arr in features]
        
        feature_matrix = np.column_stack(features)
        
        # Add derived features
        if feature_matrix.shape[1] >= 2:
            # Ratios between different test types
            ratios = []
            for i in range(feature_matrix.shape[1]):
                for j in range(i+1, feature_matrix.shape[1]):
                    ratio = feature_matrix[:, i] / (feature_matrix[:, j] + 1e-8)
                    ratios.append(ratio)
            
            if ratios:
                feature_matrix = np.column_stack([feature_matrix] + ratios)
        
        return feature_matrix
    
    def _calculate_baseline_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate baseline statistics for anomaly comparison."""
        stats = {}
        
        threshold_cols = ['pinprick_threshold', 'temp_hot_threshold', 
                         'temp_cold_threshold', 'vibration_threshold']
        
        for col in threshold_cols:
            if col in data.columns:
                values = data[col].dropna()
                stats[col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'min': values.min(),
                    'max': values.max()
                }
        
        return stats
    
    def _analyze_anomaly_types(self, data: pd.DataFrame, anomalies: np.ndarray, 
                              baseline_stats: Dict) -> Dict:
        """Analyze types of anomalies detected."""
        analysis = {
            'sensor_malfunction': [],
            'rapid_progression': [],
            'unusual_patterns': [],
            'data_quality_issues': []
        }
        
        threshold_cols = ['pinprick_threshold', 'temp_hot_threshold', 
                         'temp_cold_threshold', 'vibration_threshold']
        
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly and i < len(data):
                row = data.iloc[i]
                
                for col in threshold_cols:
                    if col in row and col in baseline_stats:
                        value = row[col]
                        stats = baseline_stats[col]
                        
                        # Check for sensor malfunction (extreme values)
                        if value > stats['max'] * 2 or value < stats['min'] * 0.5:
                            analysis['sensor_malfunction'].append({
                                'test_type': col,
                                'value': value,
                                'expected_range': [stats['min'], stats['max']]
                            })
                        
                        # Check for rapid progression (values outside 3 sigma)
                        elif abs(value - stats['mean']) > 3 * stats['std']:
                            analysis['rapid_progression'].append({
                                'test_type': col,
                                'value': value,
                                'deviation': abs(value - stats['mean']) / stats['std']
                            })
                        
                        # Unusual patterns (other anomalies)
                        else:
                            analysis['unusual_patterns'].append({
                                'test_type': col,
                                'value': value,
                                'percentile': self._calculate_percentile(value, stats)
                            })
        
        return analysis
    
    def _calculate_percentile(self, value: float, stats: Dict) -> float:
        """Calculate approximate percentile of a value."""
        if value <= stats['q25']:
            return 25.0
        elif value <= stats['median']:
            return 50.0
        elif value <= stats['q75']:
            return 75.0
        else:
            return 90.0
    def _generate_recommendations(self, anomaly_analysis: Dict) -> List[str]:
        """Generate recommendations based on anomaly analysis."""
        recommendations = []
        
        if anomaly_analysis['sensor_malfunction']:
            recommendations.append("Check sensor calibration and hardware integrity")
        
        if anomaly_analysis['rapid_progression']:
            recommendations.append("Consult healthcare provider - rapid neuropathy progression detected")
        
        if anomaly_analysis['unusual_patterns']:
            recommendations.append("Monitor closely - unusual testing patterns detected")
        
        if not any(anomaly_analysis.values()):
            recommendations.append("Continue regular monitoring schedule")
        
        return recommendations