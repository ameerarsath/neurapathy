import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, List, Tuple, Any
from functools import lru_cache
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskPredictor:
    """
    Predicts risk of foot ulceration and other diabetic complications
    based on neuropathy progression patterns.
    """
    
    def __init__(self, cache_size=128):
        self.risk_types = ['ulceration', 'amputation', 'hospitalization']
        self.models = {
            risk_type: GradientBoostingClassifier()
            for risk_type in self.risk_types
        }
        self.scalers = {}
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8
        }
        self.feature_importance = {}
        self._setup_caching(cache_size)
        
    def _setup_caching(self, cache_size):
        """Setup LRU caching for frequently accessed methods"""
        self.prepare_risk_features = lru_cache(maxsize=cache_size)(self.prepare_risk_features)
        self._get_risk_recommendation = lru_cache(maxsize=cache_size)(self._get_risk_recommendation)
        
    def prepare_risk_features(self, patient_data: pd.DataFrame, 
                            progression_data: Dict) -> np.ndarray:
        """Prepare comprehensive features for risk prediction using vectorized operations."""
        # Convert patient_data to numpy for faster operations
        data_array = patient_data.to_numpy()
        
        # Get latest thresholds using vectorized operations
        latest_thresholds = data_array[-1] if len(data_array) > 0 else np.zeros(4)
        features = list(latest_thresholds[:4])  # First 4 columns are thresholds
        
        # Add progression rates
        progression_rates = [
            progression_data.get(test_type, {}).get('progression_rate', 0)
            for test_type in ['pinprick', 'temp_hot', 'temp_cold', 'vibration']
        ]
        features.extend(progression_rates)
        
        # Calculate statistical features efficiently
        if len(data_array) >= 30:
            recent_data = data_array[-30:]
        else:
            recent_data = data_array
            
        if len(recent_data) > 0:
            # Compute all statistical features at once using numpy
            stats = np.vstack([
                np.mean(recent_data, axis=0),
                np.std(recent_data, axis=0),
                np.ptp(recent_data, axis=0),  # range
                np.percentile(recent_data, 75, axis=0) - np.percentile(recent_data, 25, axis=0)  # IQR
            ])
            features.extend(stats.flatten()[:16])  # Take first 16 features (4 stats * 4 measurements)
        else:
            features.extend([0] * 16)
        
        # Add temporal features
        if 'timestamp' in patient_data.columns:
            timestamps = pd.to_datetime(patient_data['timestamp'])
            days_since_first = (timestamps.max() - timestamps.min()).days
            features.extend([days_since_first, len(patient_data)])
        else:
            features.extend([0, 0])
        
        # Add demographic features
        features.extend([
            patient_data.get('age', 50),
            patient_data.get('diabetes_duration', 5)
        ])
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def fit(self, X: np.ndarray, y: Dict[str, np.ndarray], **params) -> None:
        """Train risk prediction models."""
        for risk_type in self.risk_types:
            risk_key = f'{risk_type}_risk'
            if risk_key in y:
                # Initialize model with parameters
                self.models[risk_type] = GradientBoostingClassifier(**params)
                # Fit the model
                self.models[risk_type].fit(X, y[risk_key])
                # Store feature importance
                self.feature_importance[risk_type] = self.models[risk_type].feature_importances_
                # Initialize scalere
                self.scalers[risk_type] = StandardScaler()
                self.scalers[risk_type].fit(X)
    
    def predict_risks(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict risk levels for all complications."""
        predictions = {}
        for risk_type in self.risk_types:
            risk_key = f'{risk_type}_risk'
            if risk_type in self.models:
                predictions[risk_key] = self.models[risk_type].predict_proba(X)[:, 1]
        return predictions
    
    def save_model(self, model_dir: str) -> None:
        """Save trained models to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for risk_type, model in self.models.items():
            model_path = model_dir / f'{risk_type}_model.pkl'
            joblib.dump(model, model_path)
    
    def load_model(self, model_dir: str) -> None:
        """Load trained models from disk."""
        model_dir = Path(model_dir)
        
        for risk_type in self.risk_types:
            model_path = model_dir / f'{risk_type}_model.pkl'
            if model_path.exists():
                self.models[risk_type] = joblib.load(model_path)
    
    def _classify_risk_level(self, probability: float) -> str:
        """Classify risk probability into risk levels."""
        if probability < self.risk_thresholds['low']:
            return 'low'
        elif probability < self.risk_thresholds['moderate']:
            return 'moderate'
        elif probability < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'critical'
    
    def _get_risk_recommendation(self, risk_type: str, risk_level: str) -> str:
        """Get recommendation based on risk type and level."""
        recommendations = {
            'ulceration_risk': {
                'low': 'Continue regular monitoring and foot care',
                'moderate': 'Increase monitoring frequency, inspect feet daily',
                'high': 'Schedule podiatrist appointment, consider protective footwear',
                'critical': 'Immediate medical attention required'
            },
            'amputation_risk': {
                'low': 'Maintain current care routine',
                'moderate': 'Enhanced foot protection and regular check-ups',
                'high': 'Immediate podiatric care and wound prevention',
                'critical': 'Emergency medical evaluation needed'
            },
            'hospitalization_risk': {
                'low': 'Continue outpatient care',
                'moderate': 'Increase medical monitoring',
                'high': 'Consider intensive outpatient program',
                'critical': 'Hospitalization may be necessary'
            }
        }
        
        return recommendations.get(risk_type, {}).get(risk_level, 'Consult healthcare provider')
    
    def _generate_priority_actions(self, risk_predictions: Dict) -> List[str]:
        """Generate prioritized action items based on all risk predictions."""
        actions = []
        
        # Sort risks by probability
        sorted_risks = sorted(risk_predictions.items(), 
                            key=lambda x: x[1]['probability'], reverse=True)
        
        for risk_type, risk_data in sorted_risks:
            if risk_data['risk_level'] in ['high', 'critical']:
                actions.append(f"Address {risk_type}: {risk_data['recommendation']}")
        
        # Add general recommendations
        if any(pred['risk_level'] == 'critical' for pred in risk_predictions.values()):
            actions.insert(0, "URGENT: Seek immediate medical attention")
        elif any(pred['risk_level'] == 'high' for pred in risk_predictions.values()):
            actions.insert(0, "Schedule medical consultation within 48 hours")
        
        return actions[:5]  # Top 5 priority actions
    
    def _calculate_feature_contributions(self, features: np.ndarray) -> Dict:
        """Calculate which features contribute most to risk prediction."""
        feature_names = [
            'pinprick_current', 'temp_hot_current', 'temp_cold_current', 'vibration_current',
            'pinprick_progression', 'temp_hot_progression', 'temp_cold_progression', 'vibration_progression',
            'pinprick_mean', 'pinprick_std', 'pinprick_range', 'pinprick_iqr',
            'temp_hot_mean', 'temp_hot_std', 'temp_hot_range', 'temp_hot_iqr',
            'temp_cold_mean', 'temp_cold_std', 'temp_cold_range', 'temp_cold_iqr',
            'vibration_mean', 'vibration_std', 'vibration_range', 'vibration_iqr',
            'days_monitored', 'total_readings', 'age', 'diabetes_duration'
        ]
        
        importances = self.feature_importance['ulceration_risk']
        
        # Calculate feature contributions
        contributions = {}
        for i, (name, importance) in enumerate(zip(feature_names, importances)):
            if i < len(features[0]):
                contributions[name] = {
                    'importance': importance,
                    'value': features[0][i],
                    'contribution': importance * abs(features[0][i])
                }
        
        # Sort by contribution
        sorted_contributions = dict(sorted(contributions.items(), 
                                         key=lambda x: x[1]['contribution'], reverse=True))
        
        return dict(list(sorted_contributions.items())[:10])  # Top 10 contributors

    def evaluate(self, X, y):
        """Evaluate model performance."""
        try:
            y_pred = self.predict_risks(X)
            y_true = y['ulceration_risk']
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro'),
                'recall_macro': recall_score(y_true, y_pred, average='macro'),
                'f1_macro': f1_score(y_true, y_pred, average='macro')
            }
            
            # Add per-class metrics
            for i, class_name in enumerate(['Low', 'Medium', 'High']):
                metrics.update({
                    f'precision_{class_name}': precision_score(y_true, y_pred, labels=[i], average='micro'),
                    f'recall_{class_name}': recall_score(y_true, y_pred, labels=[i], average='micro'),
                    f'f1_{class_name}': f1_score(y_true, y_pred, labels=[i], average='micro')
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise