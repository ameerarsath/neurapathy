# Smart Shoe ML Models

This repository contains the machine learning models and infrastructure for the Smart Shoe project.

## Project Structure

```
ml-models/                      # Root directory for all ML-related code
├── src/                       # Source code
│   ├── models/               # Model implementations
│   ├── data_preprocessing/   # Data preprocessing code
│   ├── training/            # Training code
│   └── deployment/          # API and deployment code
│
├── experiments/              # All experiment-related files
│   ├── runs/                # Experiment run outputs
│   ├── mlruns/              # MLflow tracking
│   ├── configs/             # Experiment configurations
│   └── results/             # Experiment results and analysis
│
├── data/                     # Data directory
│   ├── raw/                 # Raw input data
│   ├── processed/           # Processed data
│   └── test/                # Test datasets
│
├── trained-models/          # Saved model artifacts
│   ├── production/         # Production-ready models
│   └── experimental/       # Experimental models
│
├── tests/                   # Test files
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── data/              # Test data
│
└── docs/                    # Documentation
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
python -m pytest tests/
```

3. Train models:
```bash
python src/training/hyperparameter_tuning.py \
    --data_path "data/raw/training_data.csv" \
    --output_path "experiments/results/best_params.json" \
    --experiment_name "initial_tuning" \
    --n_trials 100
```

4. Start the API server:
```bash
uvicorn src.deployment.api_integration:app --reload
```

5. Test the API:
```bash
curl -X POST "http://localhost:8000/predict/risk" \
    -H "Content-Type: application/json" \
    -d @tests/data/test_input.json
```

## MLflow Integration

The project uses MLflow for experiment tracking and model management. MLflow data is stored in the `experiments/mlruns` directory. To view the MLflow UI:

```bash
mlflow ui --backend-store-uri file:./experiments/mlruns
```

Then visit http://localhost:5000 in your browser.

## Directory Details

- `src/`: Contains all source code
  - `models/`: Model class implementations
  - `data_preprocessing/`: Data preprocessing and feature extraction
  - `training/`: Training scripts and hyperparameter tuning
  - `deployment/`: API and deployment code

- `experiments/`: Experiment tracking and results
  - `runs/`: Individual experiment run data
  - `mlruns/`: MLflow tracking data
  - `configs/`: Configuration files for experiments
  - `results/`: Analysis results and best parameters

- `data/`: Data management
  - `raw/`: Original, unprocessed data
  - `processed/`: Cleaned and processed data
  - `test/`: Test datasets

- `trained-models/`: Model artifacts
  - `production/`: Production-ready models
  - `experimental/`: Models under development

- `tests/`: Testing infrastructure
  - `unit/`: Unit tests
  - `integration/`: Integration tests
  - `data/`: Test data files

## Contributing

1. Create a new branch for your feature
2. Write tests for new functionality
3. Ensure all tests pass
4. Submit a pull request

## License

[Add your license information here]
 