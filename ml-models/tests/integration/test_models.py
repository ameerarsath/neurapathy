import numpy as np
import pandas as pd
from src.data_preprocessing.feature_extraction import FeatureExtractor
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.deployment.api_integration import app
import uvicorn
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic sensor data
    data = {
        'patient_id': [f'P{i:03d}' for i in range(n_samples)],
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
        'pinprick_threshold': np.random.normal(10, 2, n_samples),
        'temp_hot_threshold': np.random.normal(45, 5, n_samples),
        'temp_cold_threshold': np.random.normal(15, 3, n_samples),
        'vibration_threshold': np.random.normal(25, 4, n_samples)
    }
    
    # Add synthetic risk labels
    data['ulceration_risk'] = np.random.randint(0, 2, n_samples)
    data['amputation_risk'] = np.random.randint(0, 2, n_samples)
    data['hospitalization_risk'] = np.random.randint(0, 2, n_samples)
    
    # Add synthetic progression rate
    data['progression_rate'] = np.random.normal(0.1, 0.02, n_samples)
    
    return pd.DataFrame(data)

def test_feature_extraction(data):
    """Test feature extraction pipeline."""
    logger.info("Testing feature extraction...")
    
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(data)
    
    logger.info(f"Extracted features shapes:")
    for feature_type, feature_array in features.items():
        logger.info(f"{feature_type}: {feature_array.shape}")
    
    return features

def test_model_training(features, data):
    """Test model training and hyperparameter tuning."""
    logger.info("Testing model training and hyperparameter tuning...")
    
    # Prepare target variables
    y_risk = {
        'ulceration_risk': data['ulceration_risk'].values,
        'amputation_risk': data['amputation_risk'].values,
        'hospitalization_risk': data['hospitalization_risk'].values
    }
    y_prog = data['progression_rate'].values
    
    # Initialize tuner
    tuner = HyperparameterTuner("test_experiment")
    
    # Test risk predictor optimization
    risk_params = tuner.optimize_risk_predictor(
        features['static'],  # Using static features for simplicity
        y_risk,
        n_trials=5  # Reduced for testing
    )
    logger.info(f"Best risk predictor params: {risk_params}")
    
    # Test progression tracker optimization
    prog_params = tuner.optimize_progression_tracker(
        features['temporal'],  # Using temporal features for progression
        y_prog,
        n_trials=5  # Reduced for testing
    )
    logger.info(f"Best progression tracker params: {prog_params}")
    
    # Train and save final models
    from src.models.risk_predictor import RiskPredictor
    from src.models.progression_tracker import ProgressionTracker
    
    # Train and save risk predictor
    risk_model = RiskPredictor()
    risk_model.fit(features['static'], y_risk, **risk_params)
    Path('ml-models/trained-models').mkdir(parents=True, exist_ok=True)
    risk_model.save_model('ml-models/trained-models')
    
    # Train and save progression tracker
    prog_model = ProgressionTracker()
    prog_model.fit(features['temporal'], y_prog, **prog_params)
    prog_model.save_model('ml-models/trained-models/progression-v1.pkl')
    
    return tuner

def test_api_endpoint():
    """Test API endpoint with sample data."""
    logger.info("Testing API endpoint...")
    
    # Create sample input
    test_input = {
        "patient_id": "P001",
        "timestamp": "2024-01-01T00:00:00",
        "measurements": {
            "pinprick_threshold": 10.5,
            "temp_hot_threshold": 45.2,
            "temp_cold_threshold": 15.1,
            "vibration_threshold": 25.3
        },
        "context": {
            "last_examination": "2023-12-01"
        }
    }
    
    # Save test input for API testing
    with open('test_input.json', 'w') as f:
        json.dump(test_input, f, indent=2)
    
    logger.info("Test input saved to test_input.json")
    logger.info("You can test the API using:")
    logger.info("curl -X POST http://localhost:8000/predict/risk -H 'Content-Type: application/json' -d @test_input.json")

def main():
    """Run all tests."""
    # Create data directory if it doesn't exist
    Path('data').mkdir(exist_ok=True)
    
    # Generate and save sample data
    logger.info("Generating sample data...")
    data = generate_sample_data()
    data.to_csv('data/sample_data.csv', index=False)
    logger.info("Sample data saved to data/sample_data.csv")
    
    # Run tests
    features = test_feature_extraction(data)
    tuner = test_model_training(features, data)
    
    # Save best parameters
    tuner.save_best_params('data/best_params.json')
    
    # Prepare API test
    test_api_endpoint()
    
    logger.info("All tests completed successfully!")
    logger.info("\nTo start the API server, run:")
    logger.info("uvicorn src.deployment.api_integration:app --reload")

if __name__ == "__main__":
    main() 