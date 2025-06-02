import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path so we can import our modules
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from models.baseline_model import BaselineModel

class ModelRunner:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.base_dir / "trained-models"
        self.dataset_dir = src_dir / "dataset"
        
    def load_patient_data(self, patient_id: str):
        """Load and prepare data for a specific patient"""
        try:
            # Load the relevant CSV files
            longitudinal_df = pd.read_csv(self.dataset_dir / "longitudinal_tracking.csv")
            sensor_df = pd.read_csv(self.dataset_dir / "sensor_data.csv")
            clinical_df = pd.read_csv(self.dataset_dir / "patient_clinical.csv")
            
            # Filter for the specific patient
            patient_longitudinal = longitudinal_df[longitudinal_df['patient_id'] == patient_id].sort_values('date')
            patient_sensor = sensor_df[sensor_df['patient_id'] == patient_id].sort_values('timestamp')
            patient_clinical = clinical_df[clinical_df['patient_id'] == patient_id]
            
            return patient_longitudinal, patient_sensor, patient_clinical
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def run_baseline_model(self, patient_id: str):
        """Run baseline model for a specific patient"""
        try:
            model_path = self.models_dir / f"baseline_{patient_id}.pkl"
            if not model_path.exists():
                print(f"No baseline model found for patient {patient_id}")
                return None
                
            # Load the model
            print(f"Loading model from {model_path}")
            model = BaselineModel()
            model_data = joblib.load(model_path)
            
            # Restore model state
            if isinstance(model_data, dict):
                model.scaler = model_data.get('scaler')
                model.isolation_forest = model_data.get('isolation_forest')
                model.kmeans = model_data.get('kmeans')
                model.baseline_stats = model_data.get('baseline_stats', {})
                model.is_fitted = model_data.get('is_fitted', True)
            else:
                model = model_data
            
            # Load patient data
            longitudinal, sensor, clinical = self.load_patient_data(patient_id)
            
            if longitudinal.empty:
                print(f"No longitudinal data found for patient {patient_id}")
                return None
              # Prepare features including time-based ones
            window = min(3, len(longitudinal))
            latest_data = longitudinal.iloc[-1]
            features = np.array([
                latest_data['avg_pinprick_threshold'],
                latest_data['avg_temp_hot_threshold'],
                latest_data['avg_temp_cold_threshold'],
                latest_data['avg_vibration_threshold'],
                longitudinal['avg_pinprick_threshold'].rolling(window=window).mean().iloc[-1],
                longitudinal['avg_vibration_threshold'].rolling(window=window).std().fillna(0).iloc[-1]
            ]).reshape(1, -1)
            
            # Make prediction using isolation forest
            features_scaled = model.scaler.transform(features)
            prediction = model.isolation_forest.predict(features_scaled)
            risk_score = 'High' if prediction[0] == -1 else 'Normal'
            
            # Get cluster info if available
            sensitivity_level = None
            if model.kmeans and model.is_fitted:
                cluster = model.kmeans.predict(features_scaled)[0]
                sensitivity_levels = ['Normal', 'Mild', 'Severe']
                sensitivity_level = sensitivity_levels[cluster]
            
            result = {
                'patient_id': patient_id,
                'risk_level': risk_score,
                'sensitivity_level': sensitivity_level,
                'latest_measurements': {
                    'pinprick_threshold': latest_data['avg_pinprick_threshold'],
                    'temp_hot_threshold': latest_data['avg_temp_hot_threshold'],
                    'temp_cold_threshold': latest_data['avg_temp_cold_threshold'],
                    'vibration_threshold': latest_data['avg_vibration_threshold'],
                    'response_time': latest_data['response_time_trend'],
                    'date': latest_data['date']
                },
                'baseline_stats': model.baseline_stats.get(patient_id, {})
            }
            
            # Add clinical context
            if not clinical.empty:
                result['clinical_context'] = {
                    'diabetes_type': clinical.iloc[0]['diabetes_type'],
                    'years_since_diagnosis': clinical.iloc[0]['years_since_diagnosis'],
                    'latest_hba1c': clinical.iloc[0]['latest_hba1c'],
                    'existing_neuropathy': clinical.iloc[0]['existing_neuropathy'],
                    'neuropathy_severity': clinical.iloc[0]['neuropathy_severity']
                }
            
            return result
            
        except Exception as e:
            print(f"Error running model for patient {patient_id}: {e}")
            return None
    
    def run_all_baseline_models(self):
        """Run baseline models for all patients"""
        results = []
        for model_file in self.models_dir.glob("baseline_*.pkl"):
            patient_id = model_file.stem.split('_')[1]
            result = self.run_baseline_model(patient_id)
            if result:
                results.append(result)
        return results

if __name__ == "__main__":
    runner = ModelRunner()
    
    # Example: Run for a specific patient
    print("\nTesting single patient prediction:")
    result = runner.run_baseline_model("P001")
    if result:
        print(f"\nBaseline Model Results for {result['patient_id']}:")
        print(f"Risk Level: {result['risk_level']}")
        if result['sensitivity_level']:
            print(f"Sensitivity Level: {result['sensitivity_level']}")
        print(f"\nLatest Measurements (Date: {result['latest_measurements']['date']}):")
        for metric, value in result['latest_measurements'].items():
            if metric != 'date':
                print(f"  - {metric}: {value}")
        
        print("\nBaseline Statistics:")
        stats = result['baseline_stats']
        if stats:
            print(f"  - Pinprick Threshold: {stats.get('pinprick_mean', 'N/A'):.2f} ± {stats.get('pinprick_std', 'N/A'):.2f}")
            print(f"  - Hot Temperature: {stats.get('temp_hot_mean', 'N/A'):.2f} ± {stats.get('temp_hot_std', 'N/A'):.2f}")
            print(f"  - Cold Temperature: {stats.get('temp_cold_mean', 'N/A'):.2f} ± {stats.get('temp_cold_std', 'N/A'):.2f}")
            print(f"  - Vibration: {stats.get('vibration_mean', 'N/A'):.2f} ± {stats.get('vibration_std', 'N/A'):.2f}")
            
        if 'clinical_context' in result:
            print("\nClinical Context:")
            for metric, value in result['clinical_context'].items():
                print(f"  - {metric}: {value}")
    
    # Example: Run for all patients
    print("\nTesting all patient predictions:")
    results = runner.run_all_baseline_models()
    print(f"\nProcessed {len(results)} patients:")
    for result in results:
        print(f"\nPatient {result['patient_id']}:")
        print(f"Risk Level: {result['risk_level']}")
        if result['sensitivity_level']:
            print(f"Sensitivity Level: {result['sensitivity_level']}")
        print(f"Latest Measurement Date: {result['latest_measurements']['date']}")
        print(f"Baseline Deviation: {result['risk_level']}")
