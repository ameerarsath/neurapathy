import requests
import json
from pathlib import Path

# Load test data
test_data_path = Path(__file__).parent.parent / 'data' / 'test_input.json'
with open(test_data_path, 'r') as f:
    test_data = json.load(f)

# Test risk prediction endpoint
response = requests.post(
    'http://localhost:8000/predict/risk',
    json=test_data,
    headers={'Content-Type': 'application/json'}
)

print("\nRisk Prediction Response:")
print(json.dumps(response.json(), indent=2))

# Test progression endpoint
response = requests.post(
    'http://localhost:8000/predict/progression',
    json=test_data,
    headers={'Content-Type': 'application/json'}
)

print("\nProgression Prediction Response:")
print(json.dumps(response.json(), indent=2)) 