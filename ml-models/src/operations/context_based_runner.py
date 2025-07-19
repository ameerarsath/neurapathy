import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from models.context_based_model import ContextBasedModel

class ContextBasedRunner:
    """Runner for managing individual patient context-based models"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.base_dir / "trained-models" / "context_based"
        self.dataset_dir = src_dir / "dataset"
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
    def get_patient_model(self, patient_id: str) -> Optional[ContextBasedModel]:
        """Get or create a patient-specific model"""
        model_path = self.models_dir / f"patient_{patient_id}_context.pkl"
        
        if model_path.exists():
            try:
                return ContextBasedModel.load_model(str(model_path))
            except Exception as e:
                logging.error(f"Error loading model for patient {patient_id}: {e}")
                return None
        
        # If no existing model, create new one with context
        context = self._load_patient_context(patient_id)
        if context:
            model = ContextBasedModel(patient_id)
            model.set_medical_context(context)
            return model
        return None
    
    def _load_patient_context(self, patient_id: str) -> Dict:
        """Load patient medical context from clinical data"""
        try:
            import pandas as pd
            clinical_df = pd.read_csv(self.dataset_dir / "patient_clinical.csv")
            patient_data = clinical_df[clinical_df['patient_id'] == patient_id]
            
            if patient_data.empty:
                logging.warning(f"No clinical data found for patient {patient_id}")
                return {}
                
            data = patient_data.iloc[0]
            return {
                'patient_id': patient_id,
                'age': data.get('age'),
                'diabetes_type': data.get('diabetes_type'),
                'years_since_diagnosis': data.get('years_since_diagnosis'),
                'latest_hba1c': data.get('latest_hba1c'),
                'existing_neuropathy': data.get('existing_neuropathy'),
                'neuropathy_severity': data.get('neuropathy_severity'),
                'comorbidities': data.get('comorbidities', '').split(';')
            }
            
        except Exception as e:
            logging.error(f"Error loading context for patient {patient_id}: {e}")
            return {}
    
    def process_new_reading(self, patient_id: str, reading: Dict) -> Dict:
        """Process a new sensor reading for a specific patient"""
        model = self.get_patient_model(patient_id)
        if not model:
            return {"error": f"Could not initialize model for patient {patient_id}"}
            
        try:
            # Update model with new reading
            model.update_with_new_reading(reading)
            
            # Analyze current state
            analysis = model.analyze_current_state()
            
            # Save updated model
            model.save_model(str(self.models_dir / f"patient_{patient_id}_context.pkl"))
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error processing reading for patient {patient_id}: {e}")
            return {"error": str(e)}
    
    def get_patient_analysis(self, patient_id: str) -> Dict:
        """Get current analysis for a patient"""
        model = self.get_patient_model(patient_id)
        if not model:
            return {"error": f"No model found for patient {patient_id}"}
            
        return model.analyze_current_state()

if __name__ == "__main__":
    # Example usage
    runner = ContextBasedRunner()
    
    # Example reading
    sample_reading = {
        "avg_pinprick_threshold": 3.2,
        "avg_temp_hot_threshold": 4.5,
        "avg_temp_cold_threshold": 13.8,
        "avg_vibration_threshold": 15.2,
        "response_time_trend": 420
    }
    
    # Process for a specific patient
    result = runner.process_new_reading("P001", sample_reading)
    print("\nProcessing new reading for P001:")
    print("Analysis Results:")
    for key, value in result.items():
        print(f"{key}: {value}")
        
    # Get current analysis
    analysis = runner.get_patient_analysis("P001")
    print("\nCurrent Patient Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
