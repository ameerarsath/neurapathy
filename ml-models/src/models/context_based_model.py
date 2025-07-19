import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ContextBasedModel:
    """
    A personalized model that analyzes individual patient data in isolation,
    considering their unique medical context and history.
    """
    
    def __init__(self, patient_id: str, context: Dict = None):
        self.patient_id = patient_id
        self.context = context or {}
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Patient-specific thresholds and baselines
        self.personal_baselines = {}
        self.sensitivity_thresholds = {}
        self.progression_rate = None
        self.last_update = None
        
    def set_medical_context(self, context: Dict):
        """Set patient-specific medical context"""
        required_fields = [
            'diabetes_type',
            'years_since_diagnosis',
            'existing_neuropathy',
            'latest_hba1c',
            'age',
            'comorbidities'
        ]
        
        for field in required_fields:
            if field not in context:
                logging.warning(f"Missing recommended context field: {field}")
        
        self.context = context
        self._adjust_thresholds_for_context()
    
    def _adjust_thresholds_for_context(self):
        """Adjust sensitivity thresholds based on medical context"""
        base_thresholds = {
            'pinprick': {'normal': 2.5, 'warning': 4.0, 'severe': 5.5},
            'temp_hot': {'normal': 3.5, 'warning': 6.0, 'severe': 8.5},
            'temp_cold': {'normal': 12.0, 'warning': 15.0, 'severe': 18.0},
            'vibration': {'normal': 12.0, 'warning': 20.0, 'severe': 28.0}
        }
        
        # Adjust thresholds based on patient context
        modifier = 1.0
        
        if self.context.get('age', 0) > 65:
            modifier *= 1.2
        
        if self.context.get('years_since_diagnosis', 0) > 10:
            modifier *= 1.15
            
        if self.context.get('latest_hba1c', 0) > 8.0:
            modifier *= 1.25
            
        if self.context.get('existing_neuropathy'):
            modifier *= 1.3
            
        # Apply modifiers to create personalized thresholds
        self.sensitivity_thresholds = {
            test: {level: threshold * modifier 
                  for level, threshold in thresholds.items()}
            for test, thresholds in base_thresholds.items()
        }
    
    def update_with_new_reading(self, reading: Dict):
        """Update model with new sensor reading"""
        timestamp = datetime.now()
        
        if not self.last_update:
            self.last_update = timestamp
        
        # Calculate time-based features
        time_delta = (timestamp - self.last_update).total_seconds() / 86400  # days
        
        # Extract measurements
        measurements = {
            'pinprick': reading.get('avg_pinprick_threshold'),
            'temp_hot': reading.get('avg_temp_hot_threshold'),
            'temp_cold': reading.get('avg_temp_cold_threshold'),
            'vibration': reading.get('avg_vibration_threshold')
        }
        
        # Update baselines with exponential moving average
        alpha = 0.3  # Smoothing factor
        for test, value in measurements.items():
            if value is not None:
                current = self.personal_baselines.get(test, value)
                self.personal_baselines[test] = current * (1-alpha) + value * alpha
        
        # Calculate progression rate
        if time_delta > 0:
            changes = {
                test: (value - self.personal_baselines.get(test, value)) / time_delta
                for test, value in measurements.items()
                if value is not None
            }
            self.progression_rate = np.mean(list(changes.values()))
        
        self.last_update = timestamp
        
    def analyze_current_state(self) -> Dict:
        """Analyze current patient state based on personal context and history"""
        if not self.personal_baselines:
            return {"error": "Insufficient data for analysis"}
            
        risk_factors = []
        severity_level = "Normal"
        
        # Check each measurement against personalized thresholds
        for test, baseline in self.personal_baselines.items():
            thresholds = self.sensitivity_thresholds.get(test, {})
            
            if baseline > thresholds.get('severe', float('inf')):
                risk_factors.append(f"Severe {test} sensitivity")
                severity_level = "Severe"
            elif baseline > thresholds.get('warning', float('inf')):
                risk_factors.append(f"Elevated {test} sensitivity")
                severity_level = max(severity_level, "Warning")
        
        # Consider progression rate
        if self.progression_rate:
            if self.progression_rate > 0.1:  # Rapid progression
                risk_factors.append("Rapid sensitivity progression")
                severity_level = "Severe"
            elif self.progression_rate > 0.05:  # Moderate progression
                risk_factors.append("Moderate sensitivity progression")
                severity_level = max(severity_level, "Warning")
        
        return {
            "patient_id": self.patient_id,
            "analysis_date": datetime.now().isoformat(),
            "severity_level": severity_level,
            "risk_factors": risk_factors,
            "current_baselines": self.personal_baselines,
            "progression_rate": self.progression_rate,
            "medical_context": self.context
        }
    
    def save_model(self, path: str):
        """Save the personalized model state"""
        model_state = {
            'patient_id': self.patient_id,
            'context': self.context,
            'personal_baselines': self.personal_baselines,
            'sensitivity_thresholds': self.sensitivity_thresholds,
            'progression_rate': self.progression_rate,
            'last_update': self.last_update,
            'scaler_state': self.scaler,
            'isolation_forest_state': self.isolation_forest,
            'kmeans_state': self.kmeans
        }
        joblib.dump(model_state, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'ContextBasedModel':
        """Load a saved personalized model"""
        model_state = joblib.load(path)
        model = cls(model_state['patient_id'], model_state['context'])
        
        model.personal_baselines = model_state['personal_baselines']
        model.sensitivity_thresholds = model_state['sensitivity_thresholds']
        model.progression_rate = model_state['progression_rate']
        model.last_update = model_state['last_update']
        model.scaler = model_state['scaler_state']
        model.isolation_forest = model_state['isolation_forest_state']
        model.kmeans = model_state['kmeans_state']
        
        return model
