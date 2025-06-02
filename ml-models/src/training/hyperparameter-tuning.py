import numpy as np
from typing import Dict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.info("Tuning baseline model hyperparameters...")
        return {'contamination': 0.1, 'n_estimators': 100}
    
    def tune_progression_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Tune progression tracking model."""
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        logging.info("Tuning progression model hyperparameters...")
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        logging.info(f"Best progression model params: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def tune_risk_predictor(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Tune risk prediction model."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        logging.info("Tuning risk predictor hyperparameters...")
        
        gbm = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gbm, param_grid, cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        logging.info(f"Best risk predictor params: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def save_best_params(self, filepath: str):
        """Save best parameters to file."""
        import joblib
        joblib.dump(self.best_params, filepath)
        logging.info(f"Best parameters saved to {filepath}")
    
    def load_best_params(self, filepath: str):
        """Load best parameters from file."""
        import joblib
        self.best_params = joblib.load(filepath)
        logging.info(f"Best parameters loaded from {filepath}")