from locust import HttpUser, task, between
import json
import random
from pathlib import Path

class SmartShoeUser(HttpUser):
    wait_time = between(1, 3)  # Random wait between requests
    
    def on_start(self):
        # Load test data template
        test_data_path = Path(__file__).parent.parent / 'data' / 'test_input.json'
        with open(test_data_path, 'r') as f:
            self.test_data = json.load(f)
    
    def generate_random_measurements(self):
        return {
            "pinprick_threshold": random.uniform(8, 12),
            "temp_hot_threshold": random.uniform(40, 50),
            "temp_cold_threshold": random.uniform(12, 18),
            "vibration_threshold": random.uniform(21, 29)
        }
    
    @task(2)
    def predict_risk(self):
        # Modify test data with random values
        payload = self.test_data.copy()
        payload["patient_id"] = f"P{random.randint(1, 1000000):06d}"
        payload["measurements"] = self.generate_random_measurements()
        
        self.client.post(
            "/predict/risk",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
    
    @task(1)
    def predict_progression(self):
        payload = self.test_data.copy()
        payload["patient_id"] = f"P{random.randint(1, 1000000):06d}"
        payload["measurements"] = self.generate_random_measurements()
        
        self.client.post(
            "/predict/progression",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

# To run:
# locust -f load_test.py --host=http://localhost:8000 