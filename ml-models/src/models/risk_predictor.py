import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import logging
from typing import Dict, List, Tuple

class RiskPredictor:
    """
    Predicts risk of foot ulceration and other diabetic complications
    based on neuropathy progression patterns.
    """
    
    def __init__(self):
        self.models = {
            'ulceration_risk': GradientBoostingClassifier(random_state=42),
            'amputation_risk': RandomForestClassifier(random_state=42),
            'hospitalization_risk': LogisticRegression(random_state=42)
        }
        self.scalers = {}
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8
        }
        self.feature_importance = {}
        
    def prepare_risk_features(self, patient_data: pd.DataFrame, 
                            progression_data: Dict) -> np.ndarray:
        """Prepare comprehensive features for risk prediction."""
        features = []
        
        # Neuropathy severity features
        latest_thresholds = patient_data.iloc[-1]
        features.extend([
            latest_thresholds.get('pinprick_threshold', 0),
            latest_thresholds.get('temp_hot_threshold', 0),
            latest_thresholds.get('temp_cold_threshold', 0),
            latest_thresholds.get('vibration_threshold', 0)
        ])
        
        # Progression rate features
        for test_type in ['pinprick', 'temp_hot', 'temp_cold', 'vibration']:
            if test_type in progression_data:
                features.append(progression_data[test_type].get('progression_rate', 0))
            else:
                features.append(0)
        
        # Statistical features from recent history (last 30 days)
        recent_data = patient_data.tail(30) if len(patient_data) > 30 else patient_data
        
        for col in ['pinprick_threshold', 'temp_hot_threshold', 
                   'temp_cold_threshold', 'vibration_threshold']:
            if col in recent_data.columns:
                values = recent_data[col].dropna()
                if len(values) > 0:
                    features.extend([
                        values.mean(),
                        values.std(),
                        values.max() - values.min(),  # range
                        values.quantile(0.75) - values.quantile(0.25)  # IQR
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0])
        
        # Temporal features
        if 'timestamp' in patient_data.columns:
            days_since_first = (patient_data['timestamp'].max() - 
                              patient_data['timestamp'].min()).days
            features.append(days_since_first)
            features.append(len(patient_data))  # total number of readings
        else:
            features.extend([0, 0])
        
        # Age and diabetes duration (if available)
        features.extend([
            patient_data.get('age', 50),  # default age
            patient_data.get('diabetes_duration', 5)  # default duration
        ])
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, training_data: List[Dict]) -> None:
        """
        Train risk prediction models.
        
        Args:
            training_data: List of patient data dictionaries containing:
                - patient_data: DataFrame with sensor readings
                - progression_data: Dict with progression information
                - outcomes: Dict with known outcomes (ulceration, amputation, etc.)
        """
        try:
            # Prepare training features and targets
            X = []
            y_ulceration = []
            y_amputation = []
            y_hospitalization = []
            
            for patient_record in training_data:
                # Extract features
                features = self.prepare_risk_features(
                    patient_record['patient_data'],
                    patient_record.get('progression_data', {})
                )
                X.append(features.flatten())
                
                # Extract outcomes
                outcomes = patient_record.get('outcomes', {})
                y_ulceration.append(outcomes.get('ulceration', 0))
                y_amputation.append(outcomes.get('amputation', 0))
                y_hospitalization.append(outcomes.get('hospitalization', 0))
            
            X = np.array(X)
            
            # Scale features
            for risk_type in self.models.keys():
                self.scalers[risk_type] = StandardScaler()
            
            X_scaled = self.scalers['ulceration_risk'].fit_transform(X)
            
            # Train models
            self.models['ulceration_risk'].fit(X_scaled, y_ulceration)
            self.models['amputation_risk'].fit(X_scaled, y_amputation)
            self.models['hospitalization_risk'].fit(X_scaled, y_hospitalization)
            
            # Store feature importance
            for risk_type, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[risk_type] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importance[risk_type] = np.abs(model.coef_[0])
            
            logging.info("Risk prediction models trained successfully")
            
        except Exception as e:
            logging.error(f"Error training risk models: {str(e)}")
            raise
    
    def predict_risks(self, patient_data: pd.DataFrame, 
                     progression_data: Dict, patient_id: str) -> Dict:
        """Predict various risk levels for a patient."""
        try:
            # Prepare features
            features = self.prepare_risk_features(patient_data, progression_data)
            
            risk_predictions = {}
            
            for risk_type, model in self.models.items():
                # Scale features
                features_scaled = self.scalers[risk_type].transform(features)
                
                # Predict probability
                risk_prob = model.predict_proba(features_scaled)[0, 1]
                
                # Classify risk level
                risk_level = self._classify_risk_level(risk_prob)
                
                risk_predictions[risk_type] = {
                    'probability': risk_prob,
                    'risk_level': risk_level,
                    'recommendation': self._get_risk_recommendation(risk_type, risk_level)
                }
            
            # Calculate composite risk score
            composite_score = np.mean([pred['probability'] for pred in risk_predictions.values()])
            
            return {
                'individual_risks': risk_predictions,
                'composite_risk_score': composite_score,
                'overall_risk_level': self._classify_risk_level(composite_score),
                'priority_actions': self._generate_priority_actions(risk_predictions),
                'feature_contributions': self._calculate_feature_contributions(features_scaled, risk_type='ulceration_risk')
            }
            
        except Exception as e:
            logging.error(f"Error predicting risks: {str(e)}")
            return {"error": str(e)}
    
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
    
    def _calculate_feature_contributions(self, features: np.ndarray, risk_type: str) -> Dict:
        """Calculate which features contribute most to risk prediction."""
        if risk_type not in self.feature_importance:
            return {}
        
        feature_names = [
            'pinprick_current', 'temp_hot_current', 'temp_cold_current', 'vibration_current',
            'pinprick_progression', 'temp_hot_progression', 'temp_cold_progression', 'vibration_progression',
            'pinprick_mean', 'pinprick_std', 'pinprick_range', 'pinprick_iqr',
            'temp_hot_mean', 'temp_hot_std', 'temp_hot_range', 'temp_hot_iqr',
            'temp_cold_mean', 'temp_cold_std', 'temp_cold_range', 'temp_cold_iqr',
            'vibration_mean', 'vibration_std', 'vibration_range', 'vibration_iqr',
            'days_monitored', 'total_readings', 'age', 'diabetes_duration'
        ]
        
        importances = self.feature_importance[risk_type]
        
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