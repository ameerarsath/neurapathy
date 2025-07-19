import pandas as pd
import numpy as np
from typing import Dict
from scipy.stats import pearsonr

class ClinicalCorrelationAnalyzer:
    """Analyze correlation with clinical outcomes."""
    
    def __init__(self):
        self.correlation_results = {}
        
    def analyze_clinical_correlation(self, sensor_data: pd.DataFrame, 
                                  clinical_outcomes: pd.DataFrame) -> Dict:
        """Analyze correlation between sensor readings and clinical outcomes."""
        correlations = {}
        
        merged_data = sensor_data.merge(clinical_outcomes, on='patient_id', how='inner')
        
        test_types = ['pinprick_threshold', 'temp_hot_threshold', 'temp_cold_threshold', 'vibration_threshold']
        clinical_outcomes_cols = ['ulceration', 'amputation', 'hospitalization', 'hba1c', 'neuropathy_score']
        
        for test_type in test_types:
            if test_type in merged_data.columns:
                correlations[test_type] = {}
                for outcome in clinical_outcomes_cols:
                    if outcome in merged_data.columns:
                        corr_coef = merged_data[test_type].corr(merged_data[outcome])
                        correlations[test_type][outcome] = {
                            'correlation': corr_coef,
                            'significance': self._calculate_significance(merged_data[test_type], merged_data[outcome])
                        }
        
        return correlations
    
    def validate_against_gold_standard(self, our_predictions: Dict, 
                                     gold_standard: Dict) -> Dict:
        """Validate our predictions against clinical gold standard."""
        validation_results = {}
        
        for patient_id in our_predictions.keys():
            if patient_id in gold_standard:
                our_pred = our_predictions[patient_id]
                gold_pred = gold_standard[patient_id]
                
                if 'individual_risks' in our_pred and 'clinical_risk_score' in gold_pred:
                    ulcer_risk = our_pred['individual_risks'].get('ulceration_risk', {})
                    clinical_score = gold_pred['clinical_risk_score']
                    
                    validation_results[patient_id] = {
                        'our_risk_level': ulcer_risk.get('risk_level', 'unknown'),
                        'clinical_risk_level': self._classify_clinical_risk(clinical_score),
                        'agreement': self._check_agreement(
                            ulcer_risk.get('risk_level', 'unknown'),
                            self._classify_clinical_risk(clinical_score)
                        )
                    }
        
        agreements = [r['agreement'] for r in validation_results.values()]
        overall_agreement = sum(agreements) / len(agreements) if agreements else 0
        
        return {
            'patient_comparisons': validation_results,
            'overall_agreement': overall_agreement,
            'kappa_score': self._calculate_kappa(validation_results)
        }
    
    def _calculate_significance(self, x: pd.Series, y: pd.Series) -> float:
        try:
            _, p_value = pearsonr(x.dropna(), y.dropna())
            return p_value
        except:
            return 1.0
    
    def _classify_clinical_risk(self, clinical_score: float) -> str:
        if clinical_score < 0.3:
            return 'low'
        elif clinical_score < 0.6:
            return 'moderate'
        elif clinical_score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _check_agreement(self, our_level: str, clinical_level: str) -> bool:
        return our_level == clinical_level
    
    def _calculate_kappa(self, validation_results: Dict) -> float:
        agreements = [r['agreement'] for r in validation_results.values()]
        return sum(agreements) / len(agreements) if agreements else 0