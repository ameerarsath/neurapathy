import os
import logging
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation')))
from model_validation import ModelValidator

def train_models_if_missing(models_dir: str, test_data: pd.DataFrame):
    """Train models if they don't exist."""
    from models.baseline_model import BaselineModel
    from models.progression_tracker import ProgressionTracker
    from models.risk_predictor import RiskPredictor
    
    # Change the models_dir to point to trained-models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'trained-models')
    os.makedirs(models_dir, exist_ok=True)
    
    for patient_id in test_data['patient_id'].unique():
        patient_data = test_data[test_data['patient_id'] == patient_id]
        
        # Update paths to save in trained-models directory
        baseline_path = os.path.join(models_dir, f"baseline_{patient_id}.pkl")
        progression_path = os.path.join(models_dir, f"progression_{patient_id}.pkl")
        
        # Train and save models
        if not os.path.exists(baseline_path):
            baseline_model = BaselineModel()
            baseline_model.fit(patient_data, patient_id)
            baseline_model.save_model(baseline_path)
            
        if not os.path.exists(progression_path):
            progression_model = ProgressionTracker()
            progression_model.fit(patient_data, patient_id)
            progression_model.save_model(progression_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate ML models')
    parser.add_argument('--models_dir', required=True, help='Directory containing trained models')
    parser.add_argument('--test_data', required=True, help='Path to test data CSV')
    parser.add_argument('--outcomes_data', help='Path to outcomes data CSV')
    parser.add_argument('--output_dir', required=True, help='Output directory for validation results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load test data
    logging.info("Loading test data...")
    test_data = pd.read_csv(args.test_data)
    
    # Train models if they don't exist
    logging.info("Checking and training models if needed...")
    train_models_if_missing(args.models_dir, test_data)
    
    # Initialize validator
    validator = ModelValidator(args.models_dir)
    
    # Group data by patient_id
    patient_data = {}
    for patient_id, group in test_data.groupby('patient_id'):
        patient_data[patient_id] = pd.DataFrame(group)  # Changed from dict to DataFrame
    
    # Run validations
    logging.info("Starting model validation...")
    
    # Validate baseline models
    logging.info("Validating baseline models...")
    baseline_results = validator.validate_baseline_model(patient_data)
    validator.validation_results['baseline_models'] = baseline_results
    
    # Validate progression models
    logging.info("Validating progression models...")
    progression_results = validator.validate_progression_model(patient_data)
    validator.validation_results['progression_models'] = progression_results
    
    # Prepare risk prediction data
    train_mask = test_data['weeks_monitored'] <= 2
    train_data = test_data[train_mask].copy()
    test_data_risk = test_data[~train_mask].copy()
    
    # Ensure data is 2D
    features = ['age', 'gender_encoded', 'diabetes_type_encoded', 'years_diabetes', 
                'bmi', 'hba1c_avg', 'pinprick_threshold_avg', 'temp_hot_threshold_avg',
                'temp_cold_threshold_avg', 'vibration_threshold_avg']
    
    # Validate risk prediction
    logging.info("Validating risk prediction...")
    risk_results = validator.validate_risk_predictor(
        train_data[features].values,  # Training data
        test_data_risk[features].values  # Test data
    )
    validator.validation_results['risk_prediction'] = risk_results
    
    # Generate report
    logging.info("Generating validation report...")
    os.makedirs(args.output_dir, exist_ok=True)
    validator.generate_validation_summary(args.output_dir)
    
    logging.info("Model validation completed successfully!")

if __name__ == "__main__":
    main()