import numpy as np
import pandas as pd
import sys
import os
import logging
from typing import Dict
import joblib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.progression_tracker import ProgressionTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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