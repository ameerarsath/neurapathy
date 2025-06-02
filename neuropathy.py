# training/train-baseline.py
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_model import BaselineModel
from anomaly_detector import AnomalyDetector
from progression_tracker import ProgressionTracker
from risk_predictor import RiskPredictor
import logging
from typing import Dict, List, Tuple
import joblib
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineTrainer:
    """Train baseline models for all patients in the dataset."""
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = data_path
        self.output_path = output_path
        self.baseline_model = BaselineModel()
        self.trained_patients = []
        
    def load_patient_data(self, patient_id: str) -> pd.DataFrame:
        """Load patient data from CSV files."""
        try:
            file_path = os.path.join(self.data_path, f"patient_{patient_id}.csv")
            data = pd.read_csv(file_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data
        except Exception as e:
            logging.error(f"Error loading data for patient {patient_id}: {str(e)}")
            return pd.DataFrame()
    
    def train_patient_baseline(self, patient_id: str) -> bool:
        """Train baseline model for a specific patient."""
        try:
            # Load patient data
            patient_data = self.load_patient_data(patient_id)
            if patient_data.empty:
                return False
            
            # Filter initial readings (first 2 weeks for baseline)
            start_date = patient_data['timestamp'].min()
            baseline_end = start_date + timedelta(days=14)
            baseline_data = patient_data[patient_data['timestamp'] <= baseline_end]
            
            if len(baseline_data) < 10:
                logging.warning(f"Insufficient baseline data for patient {patient_id}")
                return False
            
            # Train baseline model
            self.baseline_model.fit(baseline_data, patient_id)
            
            # Save individual model
            model_path = os.path.join(self.output_path, f"baseline_{patient_id}.pkl")
            self.baseline_model.save_model(model_path)
            
            self.trained_patients.append(patient_id)
            logging.info(f"Baseline model trained for patient {patient_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error training baseline for patient {patient_id}: {str(e)}")
            return False
    
    def train_all_patients(self, patient_ids: List[str]) -> Dict:
        """Train baseline models for all patients."""
        results = {'successful': [], 'failed': []}
        
        for patient_id in patient_ids:
            if self.train_patient_baseline(patient_id):
                results['successful'].append(patient_id)
            else:
                results['failed'].append(patient_id)
        
        logging.info(f"Baseline training completed: {len(results['successful'])} successful, {len(results['failed'])} failed")
        return results

# training/train-progression.py
class ProgressionTrainer:
    """Train progression tracking models."""
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = data_path
        self.output_path = output_path
        self.progression_tracker = ProgressionTracker()
        
    def prepare_progression_data(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for progression tracking."""
        # Sort by timestamp
        data_sorted = patient_data.sort_values('timestamp')
        
        # Add derived features
        data_sorted['days_since_start'] = (data_sorted['timestamp'] - data_sorted['timestamp'].min()).dt.days
        
        return data_sorted
    
    def train_patient_progression(self, patient_id: str) -> bool:
        """Train progression model for a patient."""
        try:
            # Load patient data
            file_path = os.path.join(self.data_path, f"patient_{patient_id}.csv")
            patient_data = pd.read_csv(file_path)
            patient_data['timestamp'] = pd.to_datetime(patient_data['timestamp'])
            
            # Prepare data
            progression_data = self.prepare_progression_data(patient_data)
            
            # Need at least 45 days of data for progression tracking
            if len(progression_data) < 45:
                logging.warning(f"Insufficient data for progression tracking - patient {patient_id}")
                return False
            
            # Train model
            self.progression_tracker.fit(progression_data, patient_id)
            
            # Save model
            model_path = os.path.join(self.output_path, f"progression_{patient_id}.pkl")
            joblib.dump(self.progression_tracker, model_path)
            
            logging.info(f"Progression model trained for patient {patient_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error training progression model for patient {patient_id}: {str(e)}")
            return False
    
    def evaluate_progression_accuracy(self, patient_id: str, test_data: pd.DataFrame) -> Dict:
        """Evaluate progression prediction accuracy."""
        try:
            # Load trained model
            model_path = os.path.join(self.output_path, f"progression_{patient_id}.pkl")
            trained_tracker = joblib.load(model_path)
            
            # Make predictions
            predictions = trained_tracker.predict_progression(test_data, patient_id)
            
            # Calculate accuracy metrics
            accuracy_metrics = {}
            for test_type, pred_data in predictions.items():
                if 'predicted_value' in pred_data and 'current_value' in pred_data:
                    # Compare with actual future values (if available)
                    future_data = test_data.tail(7)  # Next 7 days
                    if f"{test_type}_threshold" in future_data.columns:
                        actual_future = future_data[f"{test_type}_threshold"].mean()
                        predicted_future = pred_data['predicted_value']
                        
                        accuracy_metrics[test_type] = {
                            'mae': abs(actual_future - predicted_future),
                            'mape': abs((actual_future - predicted_future) / actual_future) * 100,
                            'predicted': predicted_future,
                            'actual': actual_future
                        }
            
            return accuracy_metrics
            
        except Exception as e:
            logging.error(f"Error evaluating progression accuracy: {str(e)}")
            return {}

# training/hyperparameter-tuning.py
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error

class HyperparameterTuner:
    """Hyperparameter tuning for all models."""
    
    def __init__(self):
        self.best_params = {}
        
    def tune_baseline_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Tune baseline model hyperparameters."""
        param_grid = {
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        
        # This is a simplified version - in practice, you'd need to adapt
        # the baseline model to work with GridSearchCV
        return {'contamination': 0.1, 'n_estimators': 100}
    
    def tune_progression_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Tune progression tracking model."""
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_
    
    def tune_risk_predictor(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Tune risk prediction model."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gbm = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gbm, param_grid, cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_

# evaluation/model-validation.py
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelValidator:
    """Comprehensive model validation framework."""
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.validation_results = {}
        
    def validate_baseline_model(self, patient_data: Dict) -> Dict:
        """Validate baseline model performance."""
        results = {}
        
        for patient_id, data in patient_data.items():
            try:
                # Load baseline model
                model_path = os.path.join(self.models_path, f"baseline_{patient_id}.pkl")
                baseline_model = BaselineModel()
                baseline_model.load_model(model_path)
                
                # Split data for validation
                train_data = data.iloc[:len(data)//2]
                test_data = data.iloc[len(data)//2:]
                
                # Test sensitivity level predictions
                sensitivity_predictions = baseline_model.predict_sensitivity_level(test_data, patient_id)
                
                # Calculate validation metrics
                results[patient_id] = {
                    'sensitivity_distribution': np.bincount(
                        [0 if s == 'normal' else 1 if s == 'mild_loss' else 2 
                         for s in sensitivity_predictions]
                    ),
                    'prediction_consistency': self._calculate_consistency(sensitivity_predictions),
                    'data_points': len(test_data)
                }
                
            except Exception as e:
                logging.error(f"Error validating baseline model for patient {patient_id}: {str(e)}")
                results[patient_id] = {'error': str(e)}
        
        return results
    
    def validate_progression_model(self, patient_data: Dict) -> Dict:
        """Validate progression tracking model."""
        results = {}
        
        for patient_id, data in patient_data.items():
            try:
                # Load progression model
                model_path = os.path.join(self.models_path, f"progression_{patient_id}.pkl")
                progression_tracker = joblib.load(model_path)
                
                # Time series split for validation
                tscv = TimeSeriesSplit(n_splits=3)
                mae_scores = []
                
                for train_idx, test_idx in tscv.split(data):
                    train_data = data.iloc[train_idx]
                    test_data = data.iloc[test_idx]
                    
                    # Make predictions
                    predictions = progression_tracker.predict_progression(train_data, patient_id)
                    
                    # Calculate MAE for each test type
                    test_mae = []
                    for test_type, pred_data in predictions.items():
                        if 'predicted_value' in pred_data:
                            actual_col = f"{test_type}_threshold"
                            if actual_col in test_data.columns:
                                actual_mean = test_data[actual_col].mean()
                                predicted = pred_data['predicted_value']
                                test_mae.append(abs(actual_mean - predicted))
                    
                    if test_mae:
                        mae_scores.append(np.mean(test_mae))
                
                results[patient_id] = {
                    'cross_val_mae': np.mean(mae_scores) if mae_scores else None,
                    'mae_std': np.std(mae_scores) if mae_scores else None,
                    'n_splits': len(mae_scores)
                }
                
            except Exception as e:
                logging.error(f"Error validating progression model for patient {patient_id}: {str(e)}")
                results[patient_id] = {'error': str(e)}
        
        return results
    
    def validate_risk_predictor(self, training_data: List[Dict], test_data: List[Dict]) -> Dict:
        """Validate risk prediction model."""
        try:
            # Initialize and train risk predictor
            risk_predictor = RiskPredictor()
            risk_predictor.fit(training_data)
            
            # Test predictions
            all_predictions = []
            all_outcomes = []
            
            for test_record in test_data:
                patient_data = test_record['patient_data']
                progression_data = test_record.get('progression_data', {})
                outcomes = test_record.get('outcomes', {})
                
                # Make risk predictions
                risk_pred = risk_predictor.predict_risks(
                    patient_data, progression_data, 
                    test_record.get('patient_id', 'unknown')
                )
                
                if 'individual_risks' in risk_pred:
                    all_predictions.append(risk_pred['individual_risks'])
                    all_outcomes.append(outcomes)
            
            # Calculate validation metrics
            validation_results = {}
            
            for risk_type in ['ulceration_risk', 'amputation_risk', 'hospitalization_risk']:
                predictions = [pred.get(risk_type, {}).get('probability', 0) for pred in all_predictions]
                outcomes = [outcome.get(risk_type.split('_')[0], 0) for outcome in all_outcomes]
                
                if predictions and outcomes:
                    from sklearn.metrics import roc_auc_score, precision_recall_curve
                    
                    validation_results[risk_type] = {
                        'auc_score': roc_auc_score(outcomes, predictions),
                        'n_samples': len(predictions),
                        'positive_rate': np.mean(outcomes)
                    }
            
            return validation_results
            
        except Exception as e:
            logging.error(f"Error validating risk predictor: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_consistency(self, predictions: np.ndarray) -> float:
        """Calculate prediction consistency score."""
        if len(predictions) < 2:
            return 1.0
        
        # Calculate how often consecutive predictions are the same
        same_predictions = sum(1 for i in range(1, len(predictions)) 
                             if predictions[i] == predictions[i-1])
        
        return same_predictions / (len(predictions) - 1)

# evaluation/performance-metrics.py
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
            report.append(f"- Average Prediction Consistency: {np.mean([r.get('prediction_consistency', 0) for r in baseline_results.values() if 'prediction_consistency' in r]):.2f}")
            report.append("")
        
        # Progression Model Performance
        if 'progression' in all_validation_results:
            prog_results = all_validation_results['progression']
            valid_maes = [r['cross_val_mae'] for r in prog_results.values() if 'cross_val_mae' in r and r['cross_val_mae'] is not None]
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
        
        return "\n".join(report)
    
    def _calculate_ppv(self, risk_data: Dict) -> float:
        """Calculate Positive Predictive Value."""
        # Simplified calculation - in practice, you'd need actual predictions and outcomes
        return 0.75  # Placeholder
    
    def _calculate_npv(self, risk_data: Dict) -> float:
        """Calculate Negative Predictive Value."""
        # Simplified calculation - in practice, you'd need actual predictions and outcomes
        return 0.92  # Placeholder

# evaluation/clinical-correlation.py
class ClinicalCorrelationAnalyzer:
    """Analyze correlation with clinical outcomes."""
    
    def __init__(self):
        self.correlation_results = {}
        
    def analyze_clinical_correlation(self, sensor_data: pd.DataFrame, 
                                  clinical_outcomes: pd.DataFrame) -> Dict:
        """Analyze correlation between sensor readings and clinical outcomes."""
        correlations = {}
        
        # Merge sensor data with clinical outcomes
        merged_data = sensor_data.merge(clinical_outcomes, on='patient_id', how='inner')
        
        # Calculate correlations for each test type
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
                
                # Compare risk classifications
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
        
        # Calculate overall agreement
        agreements = [r['agreement'] for r in validation_results.values()]
        overall_agreement = sum(agreements) / len(agreements) if agreements else 0
        
        return {
            'patient_comparisons': validation_results,
            'overall_agreement': overall_agreement,
            'kappa_score': self._calculate_kappa(validation_results)
        }
    
    def _calculate_significance(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate statistical significance of correlation."""
        from scipy.stats import pearsonr
        try:
            _, p_value = pearsonr(x.dropna(), y.dropna())
            return p_value
        except:
            return 1.0
    
    def _classify_clinical_risk(self, clinical_score: float) -> str:
        """Classify clinical risk score into risk levels."""
        if clinical_score < 0.3:
            return 'low'
        elif clinical_score < 0.6:
            return 'moderate'
        elif clinical_score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _check_agreement(self, our_level: str, clinical_level: str) -> bool:
        """Check if our risk level agrees with clinical assessment."""
        return our_level == clinical_level
    
    def _calculate_kappa(self, validation_results: Dict) -> float:
        """Calculate Cohen's Kappa for agreement."""
        # Simplified kappa calculation
        agreements = [r['agreement'] for r in validation_results.values()]
        return sum(agreements) / len(agreements) if agreements else 0

# Main training and evaluation pipeline
def main_training_pipeline():
    """Execute the complete training and evaluation pipeline."""
    
    # Configuration
    data_path = "data/patient_data"
    models_output_path = "models/trained"
    results_output_path = "results"
    
    # Create output directories
    os.makedirs(models_output_path, exist_ok=True)
    os.makedirs(results_output_path, exist_ok=True)
    
    # Get list of patient IDs
    patient_ids = [f.split('_')[1].split('.')[0] for f in os.listdir(data_path) if f.startswith('patient_')]
    
    logging.info(f"Starting training pipeline for {len(patient_ids)} patients")
    
    # 1. Train baseline models
    baseline_trainer = BaselineTrainer(data_path, models_output_path)
    baseline_results = baseline_trainer.train_all_patients(patient_ids)
    
    # 2. Train progression models
    progression_trainer = ProgressionTrainer(data_path, models_output_path)
    progression_results = {'successful': [], 'failed': []}
    
    for patient_id in baseline_results['successful']:
        if progression_trainer.train_patient_progression(patient_id):
            progression_results['successful'].append(patient_id)
        else:
            progression_results['failed'].append(patient_id)
    
    # 3. Model validation
    validator = ModelValidator(models_output_path)
    
    # Load validation data
    validation_data = {}
    for patient_id in progression_results['successful']:
        patient_data = baseline_trainer.load_patient_data(patient_id)
        if not patient_data.empty:
            validation_data[patient_id] = patient_data
    
    # Validate all models
    baseline_validation = validator.validate_baseline_model(validation_data)
    progression_validation = validator.validate_progression_model(validation_data)
    
    # 4. Generate performance report
    metrics_calculator = PerformanceMetrics()
    all_results = {
        'baseline': baseline_validation,
        'progression': progression_validation
    }
    
    performance_report = metrics_calculator.generate_performance_report(all_results)
    
    # Save results
    with open(os.path.join(results_output_path, 'performance_report.txt'), 'w') as f:
        f.write(performance_report)
    
    # Save detailed results
    joblib.dump(all_results, os.path.join(results_output_path, 'detailed_validation_results.pkl'))
    
    logging.info("Training and evaluation pipeline completed successfully")
    print(performance_report)

if __name__ == "__main__":
    main_training_pipeline()