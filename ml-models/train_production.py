import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split
from src.data_preprocessing.feature_extraction import FeatureExtractor
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.models.risk_predictor import RiskPredictor
from src.models.progression_tracker import ProgressionTracker
from src.data_preprocessing.production_feature_extractor import ProductionFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up MLflow to store everything within ml-models directory
MLFLOW_TRACKING_URI = str(Path(__file__).parent / "experiments" / "mlruns")
mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_URI}")
mlflow.set_experiment("smart_shoe_production")

class ProductionTrainer:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.root_dir = Path(__file__).parent
        
        # Ensure all directories exist
        (self.root_dir / "experiments" / "mlruns").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "experiments" / "results").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "experiments" / "configs").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "trained-models").mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = ProductionFeatureExtractor()
        self.tuner = HyperparameterTuner("smart_shoe_production")

    def train_model(self, data_params, model_params):
        """Train the model with MLflow tracking"""
        with mlflow.start_run(run_name="training_run") as run:
            try:
                # Log parameters
                mlflow.log_params(data_params)
                mlflow.log_params(model_params)
                
                # Load and preprocess data
                logger.info("Loading and preprocessing data...")
                X_train, X_test, y_train, y_test = self._prepare_data(data_params)
                
                # Train model
                logger.info("Training model...")
                model = self._train_model(X_train, y_train, model_params)
                
                # Evaluate model
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Log metrics
                mlflow.log_metric("train_accuracy", train_score)
                mlflow.log_metric("test_accuracy", test_score)
                
                # Save results
                results = {
                    "run_id": run.info.run_id,
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                    "params": {**data_params, **model_params}
                }
                results_path = self.root_dir / "experiments" / "results" / f"run_{run.info.run_id}.json"
                with open(results_path, 'w') as f:
                    import json
                    json.dump(results, f, indent=4)
                
                # Save model and log to MLflow
                model_path = self.output_dir / f"model_{run.info.run_id}.pkl"
                mlflow.sklearn.save_model(model, model_path)
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    registered_model_name="smart_shoe_model"
                )
                
                # Log feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self._log_feature_importance(model, X_train.columns)
                
                logger.info(f"Training completed. Model saved to {model_path}")
                logger.info(f"Results saved to {results_path}")
                logger.info(f"Run ID: {run.info.run_id}")
                
                return model, test_score
                
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                mlflow.log_param("error", str(e))
                raise

    def _prepare_data(self, data_params):
        """Prepare data for training"""
        try:
            # Use sample data if sensor_data.csv doesn't exist
            data_path = self.data_dir / "sample_data.csv"
            logger.info(f"Loading data from {data_path}")
            
            # Read the data
            df = pd.read_csv(data_path)
            
            # Extract features using the feature extractor
            X = self.feature_extractor.extract_features(df)
            
            # Create target variable (multi-target classification)
            y = df[['ulceration_risk', 'amputation_risk', 'hospitalization_risk']].values
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=data_params.get("test_size", 0.2),
                random_state=data_params.get("random_state", 42)
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def _train_model(self, X_train, y_train, model_params):
        """Train the model"""
        try:
            from sklearn.multioutput import MultiOutputClassifier
            from sklearn.ensemble import RandomForestClassifier
            
            # Create base classifier
            base_clf = RandomForestClassifier(
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", 10),
                min_samples_split=model_params.get("min_samples_split", 5),
                random_state=42
            )
            
            # Create multi-output classifier
            model = MultiOutputClassifier(base_clf)
            
            # Train the model
            model.fit(X_train, y_train)
            return model
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def _log_feature_importance(self, model, feature_names):
        """Log feature importance plot to MLflow"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        importances = pd.DataFrame({
            'features': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.bar(importances['features'], importances['importance'])
        plt.xticks(rotation=45)
        plt.title('Feature Importances')
        
        # Save plot to temp file and log to MLflow
        plot_path = "feature_importance.png"
        plt.savefig(plot_path, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)
    
    def _log_sample_input(self, sample_input):
        """Log sample input for model inference"""
        sample_input_dict = sample_input.to_dict()
        mlflow.log_dict(sample_input_dict, "sample_input.json")

# Example usage
if __name__ == "__main__":
    data_params = {
        "data_path": "raw/sensor_data.csv",
        "test_size": 0.2,
        "random_state": 42
    }
    
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5
    }
    
    trainer = ProductionTrainer("data", "trained-models")
    model, score = trainer.train_model(data_params, model_params) 