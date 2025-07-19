import numpy as np
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PerformanceMetrics:
    """Calculate comprehensive performance metrics."""
    
    def __init__(self):
        self.metrics_history = {}
        
    def calculate_clinical_metrics(self, validation_results: Dict) -> Dict:
        """Calculate clinically relevant metrics."""
        clinical_metrics = {}
        
        # Sensitivity and Specificity for risk prediction
        if 'ulceration_risk' in validation_results:
            ulcer_data = validation_results['ulceration_risk']
            clinical_metrics['ulceration_detection'] = {
                'auc': ulcer_data.get('auc_score', 0),
                'clinical_threshold': 0.3,  # Clinical decision threshold
                'positive_predictive_value': self._calculate_ppv(ulcer_data),
                'negative_predictive_value': self._calculate_npv(ulcer_data)
            }
        
        # Progression detection accuracy
        progression_maes = []
        for patient_id, results in validation_results.items():
            if isinstance(results, dict) and 'cross_val_mae' in results:
                if results['cross_val_mae'] is not None:
                    progression_maes.append(results['cross_val_mae'])
        
        if progression_maes:
            clinical_metrics['progression_tracking'] = {
                'mean_absolute_error': np.mean(progression_maes),
                'error_std': np.std(progression_maes),
                'clinical_significance_threshold': 0.15,  # 15% change threshold
                'accuracy_within_threshold': sum(1 for mae in progression_maes if mae < 0.15) / len(progression_maes)
            }
        
        return clinical_metrics
    
    def generate_performance_report(self, all_validation_results: Dict) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("=== DIABETIC NEUROPATHY MONITORING SYSTEM - PERFORMANCE REPORT ===\n")
        
        # Baseline Model Performance
        if 'baseline' in all_validation_results:
            baseline_results = all_validation_results['baseline']
            report.append("BASELINE MODEL PERFORMANCE:")
            report.append(f"- Patients Successfully Modeled: {len([r for r in baseline_results.values() if 'error' not in r])}")
            
            consistency_scores = [r.get('prediction_consistency', 0) for r in baseline_results.values() 
                                if 'prediction_consistency' in r]
            if consistency_scores:
                report.append(f"- Average Prediction Consistency: {np.mean(consistency_scores):.2f}")
            report.append("")
        
        # Progression Model Performance
        if 'progression' in all_validation_results:
            prog_results = all_validation_results['progression']
            valid_maes = [r['cross_val_mae'] for r in prog_results.values() 
                         if 'cross_val_mae' in r and r['cross_val_mae'] is not None]
            if valid_maes:
                report.append("PROGRESSION TRACKING PERFORMANCE:")
                report.append(f"- Mean Absolute Error: {np.mean(valid_maes):.3f}")
                report.append(f"- Error Standard Deviation: {np.std(valid_maes):.3f}")
                report.append(f"- Patients with Reliable Tracking: {len(valid_maes)}")
                report.append("")
        
        # Risk Prediction Performance
        if 'risk' in all_validation_results:
            risk_results = all_validation_results['risk']
            report.append("RISK PREDICTION PERFORMANCE:")
            for risk_type, metrics in risk_results.items():
                if isinstance(metrics, dict) and 'auc_score' in metrics:
                    report.append(f"- {risk_type.replace('_', ' ').title()}: AUC = {metrics['auc_score']:.3f}")
            report.append("")
        
        # Clinical Significance
        clinical_metrics = self.calculate_clinical_metrics(all_validation_results.get('risk', {}))
        if clinical_metrics:
            report.append("CLINICAL SIGNIFICANCE:")
            if 'ulceration_detection' in clinical_metrics:
                ulcer_metrics = clinical_metrics['ulceration_detection']
                report.append(f"- Ulceration Risk AUC: {ulcer_metrics['auc']:.3f}")
                report.append(f"- Clinical Decision Threshold: {ulcer_metrics['clinical_threshold']}")
            
            if 'progression_tracking' in clinical_metrics:
                prog_metrics = clinical_metrics['progression_tracking']
                report.append(f"- Progression Tracking Accuracy: {prog_metrics['accuracy_within_threshold']:.1%}")
                report.append(f"- Mean Absolute Error: {prog_metrics['mean_absolute_error']:.3f}")
        
        report.append("\n" + "="*70)
        return "\n".join(report)
    
    def calculate_model_performance_summary(self, validation_results: Dict) -> Dict:
        """Calculate summary statistics for model performance."""
        summary = {
            'total_patients': 0,
            'successful_models': 0,
            'failed_models': 0,
            'average_accuracy': 0.0,
            'performance_distribution': {}
        }
        
        if 'baseline' in validation_results:
            baseline_results = validation_results['baseline']
            summary['total_patients'] = len(baseline_results)
            summary['successful_models'] = len([r for r in baseline_results.values() if 'error' not in r])
            summary['failed_models'] = summary['total_patients'] - summary['successful_models']
        
        return summary
    
    def _calculate_ppv(self, risk_data: Dict) -> float:
        """Calculate Positive Predictive Value."""
        # Simplified calculation - in practice, you'd need actual predictions and outcomes
        return 0.75  # Placeholder
    
    def _calculate_npv(self, risk_data: Dict) -> float:
        """Calculate Negative Predictive Value."""
        # Simplified calculation - in practice, you'd need actual predictions and outcomes
        return 0.92  # Placeholder
    
    def save_metrics_history(self, filepath: str):
        """Save metrics history to file."""
        import joblib
        joblib.dump(self.metrics_history, filepath)
        logging.info(f"Metrics history saved to {filepath}")
    
    def load_metrics_history(self, filepath: str):
        """Load metrics history from file."""
        import joblib
        self.metrics_history = joblib.load(filepath)  
        logging.info(f"Metrics history loaded from {filepath}")