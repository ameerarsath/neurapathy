import numpy as np
import pandas as pd
import sys
import os
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_model import BaselineModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineTrainer:
    """Train baseline models for all patients in the dataset."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "dataset"
        self.models_dir = self.base_dir.parent / "trained-models"
        self.models_dir.mkdir(exist_ok=True)
        self.trained_patients = []
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all dataset files."""
        longitudinal_df = pd.read_csv(self.data_dir / "longitudinal_tracking.csv")
        sensor_df = pd.read_csv(self.data_dir / "sensor_data.csv")
        clinical_df = pd.read_csv(self.data_dir / "patient_clinical.csv")
        
        # Convert date/timestamp columns
        longitudinal_df['date'] = pd.to_datetime(longitudinal_df['date'])
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
        
        return longitudinal_df, sensor_df, clinical_df
    
    def prepare_patient_data(self, patient_id: str, 
                           longitudinal_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare training data for a single patient."""
        # Get longitudinal data for the patient
        patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
        patient_data.sort_values('date', inplace=True)
        
        if len(patient_data) < 3:
            logging.warning(f"Insufficient longitudinal data for patient {patient_id}")
            return pd.DataFrame()
        
        # Create feature matrix with both static and time-based features
        features = pd.DataFrame({
            'pinprick_threshold_avg': patient_data['avg_pinprick_threshold'],
            'temp_hot_threshold_avg': patient_data['avg_temp_hot_threshold'],
            'temp_cold_threshold_avg': patient_data['avg_temp_cold_threshold'],
            'vibration_threshold_avg': patient_data['avg_vibration_threshold'],
            'pinprick_rolling_mean': patient_data['avg_pinprick_threshold'].rolling(window=3).mean().fillna(patient_data['avg_pinprick_threshold']),
            'vibration_rolling_std': patient_data['avg_vibration_threshold'].rolling(window=3).std().fillna(0)
        })
        
        return features
    
    def train_patient_model(self, patient_id: str,
                          longitudinal_df: pd.DataFrame) -> bool:
        """Train baseline model for a single patient."""
        try:
            # Prepare training data
            training_data = self.prepare_patient_data(patient_id, longitudinal_df)
            
            if len(training_data) < 3:
                return False
            
            # Create and train model
            model = BaselineModel()
            model.fit(training_data, patient_id)
            
            # Save trained model
            model_path = self.models_dir / f"baseline_{patient_id}.pkl"
            model_state = {
                'scaler': model.scaler,
                'isolation_forest': model.isolation_forest,
                'kmeans': model.kmeans,
                'baseline_stats': model.baseline_stats,
                'is_fitted': model.is_fitted
            }
            joblib.dump(model_state, model_path)
            
            self.trained_patients.append(patient_id)
            logging.info(f"Successfully trained and saved model for patient {patient_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error training model for patient {patient_id}: {str(e)}")
            return False
    
    def train_all_models(self) -> None:
        """Train baseline models for all patients in the dataset."""
        # Load all data
        longitudinal_df, sensor_df, clinical_df = self.load_data()
        
        # Get unique patient IDs with sufficient data
        patient_ids = longitudinal_df['patient_id'].unique()
        
        success_count = 0
        for patient_id in patient_ids:
            if self.train_patient_model(patient_id, longitudinal_df):
                success_count += 1
                
        logging.info(f"Training complete. Successfully trained {success_count}/{len(patient_ids)} models.")
        
if __name__ == "__main__":
    trainer = BaselineTrainer()
    trainer.train_all_models()