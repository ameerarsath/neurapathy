import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, accuracy_score
import optuna
import mlflow
import logging
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.baseline_model import BaselineModel
from models.progression_tracker import ProgressionTracker
from models.risk_predictor import RiskPredictor

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Optimize model hyperparameters using Optuna and MLflow tracking."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.best_params = {}
        self.best_scores = {}
        
    def optimize_risk_predictor(self, X: np.ndarray, y: Dict[str, np.ndarray],
                              n_trials: int = 100) -> Dict[str, Any]:
        """Optimize Risk Predictor hyperparameters."""
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            
            try:
                # Initialize model
                model = RiskPredictor()
                
                # Evaluate using cross-validation
                scores = []
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
                for train_idx, val_idx in cv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = {k: v[train_idx] for k, v in y.items()}
                    y_val = {k: v[val_idx] for k, v in y.items()}
                    
                    # Train model with parameters
                    model.fit(X_train, y_train, **params)
                    
                    # Evaluate
                    y_pred = model.predict_risks(X_val)
                    # Calculate average accuracy across all risk types
                    accuracies = []
                    for risk_type in model.risk_types:
                        risk_key = f'{risk_type}_risk'
                        if risk_key in y_val and risk_key in y_pred:
                            pred_labels = (y_pred[risk_key] > 0.5).astype(int)
                            accuracies.append(accuracy_score(y_val[risk_key], pred_labels))
                    score = np.mean(accuracies) if accuracies else 0.0
                    scores.append(score)
                
                mean_score = np.mean(scores)
                
                # Log to MLflow
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric('mean_cv_accuracy', mean_score)
                
                return mean_score
                
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        if study.best_trial is not None:
            self.best_params['risk_predictor'] = study.best_params
            self.best_scores['risk_predictor'] = study.best_value
            logger.info(f"Best risk predictor params: {study.best_params}")
            return study.best_params
        else:
            logger.warning("No successful trials found")
            return {}
    
    def optimize_progression_tracker(self, X: np.ndarray, y: Dict[str, np.ndarray],
                                   n_trials: int = 100) -> Dict[str, Any]:
        """Optimize Progression Tracker hyperparameters."""
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            
            try:
                # Initialize model
                model = ProgressionTracker()
                
                # Evaluate using cross-validation
                scores = []
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
                for train_idx, val_idx in cv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model with parameters
                    model.fit(X_train, y_train, **params)
                    
                    # Evaluate
                    y_pred = model.predict_progression(X_val)
                    score = -mean_squared_error(y_val, y_pred)
                    scores.append(score)
                
                mean_score = np.mean(scores)
                
                # Log to MLflow
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric('mean_cv_neg_mse', mean_score)
                
                return mean_score
                
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        if study.best_trial is not None:
            self.best_params['progression_tracker'] = study.best_params
            self.best_scores['progression_tracker'] = study.best_value
            logger.info(f"Best progression tracker params: {study.best_params}")
            return study.best_params
        else:
            logger.warning("No successful trials found")
            return {}
        
    def optimize_baseline_model(self, X: np.ndarray, y: np.ndarray,
                              n_trials: int = 100) -> Dict[str, Any]:
        """Optimize Baseline Model hyperparameters."""
        
        def objective(trial):
            params = {
                'contamination': trial.suggest_float('contamination', 0.01, 0.2),
                'n_clusters': trial.suggest_int('n_clusters', 2, 5),
                'isolation_estimators': trial.suggest_int('isolation_estimators', 50, 200)
            }
            
            model = BaselineModel()
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model.fit(X_train, y_train, params)
                
                # Evaluate
                predictions = model.predict_sensitivity_level(X_val)
                score = np.mean(predictions == y_val)
                scores.append(score)
            
            mean_score = np.mean(scores)
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric('mean_accuracy', mean_score)
            
            return mean_score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['baseline_model'] = study.best_params
        self.best_scores['baseline_model'] = study.best_value
        
        return study.best_params
    
    def save_best_params(self, output_path: str) -> None:
        """Save best hyperparameters to file."""
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'best_parameters': self.best_params,
            'best_scores': self.best_scores
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Best parameters saved to {output_path}")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--data_path', required=True, help='Path to training data')
    parser.add_argument('--output_path', required=True, help='Path to save results')
    parser.add_argument('--experiment_name', default='hyperparameter_tuning',
                       help='MLflow experiment name')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of optimization trials')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    data = pd.read_csv(args.data_path)
    
    # Initialize tuner
    tuner = HyperparameterTuner(args.experiment_name)
    
    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    
    # Optimize all models
    tuner.optimize_risk_predictor(data['X'].values, data['y_risk'].values,
                                n_trials=args.n_trials)
    tuner.optimize_progression_tracker(data['X'].values, data['y_prog'].values,
                                    n_trials=args.n_trials)
    tuner.optimize_baseline_model(data['X'].values, data['y_base'].values,
                                n_trials=args.n_trials)
    
    # Save results
    tuner.save_best_params(args.output_path)
    
    logger.info("Hyperparameter optimization completed successfully") 