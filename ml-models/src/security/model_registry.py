"""
Model registry for secure model management and tracking.
"""
import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import asdict

from .security_config import SecurityConfig, ModelMetadata
from .enhanced_security import ModelSecurity
from .audit import SecureAuditLogger

class ModelRegistry:
    """Central registry for managing and tracking ML models."""
    
    def __init__(self, 
                 registry_dir: str,
                 security_config: Optional[SecurityConfig] = None):
        """Initialize model registry."""
        self.registry_dir = registry_dir
        self.config = security_config or SecurityConfig()
        self.security = ModelSecurity(
            log_dir=os.path.join(registry_dir, 'logs')
        )
        self.audit = self.security.audit
        
        # Create registry structure
        os.makedirs(registry_dir, exist_ok=True)
        os.makedirs(os.path.join(registry_dir, 'metadata'), exist_ok=True)
        os.makedirs(os.path.join(registry_dir, 'checksums'), exist_ok=True)
        
        self.metadata_file = os.path.join(registry_dir, 'metadata', 'model_metadata.json')
        self.checksums_file = os.path.join(registry_dir, 'checksums', 'model_checksums.json')
        self._init_registry()
        
    def _init_registry(self) -> None:
        """Initialize registry files if they don't exist."""
        if not os.path.exists(self.metadata_file):
            self._save_metadata({})
        if not os.path.exists(self.checksums_file):
            self._save_checksums({
                'checksums': {},
                'last_updated': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            })
            
    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _save_checksums(self, checksums: Dict) -> None:
        """Save checksums to file."""
        with open(self.checksums_file, 'w') as f:
            json.dump(checksums, f, indent=2)
            
    def register_model(self, 
                      model_path: str,
                      metadata: ModelMetadata) -> None:
        """Register a new model with the registry."""
        try:
            # Verify model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            # Load existing metadata
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
                
            # Update metadata
            all_metadata[metadata.model_id] = asdict(metadata)
            self._save_metadata(all_metadata)
            
            # Update checksums
            with open(self.checksums_file, 'r') as f:
                checksums = json.load(f)
                
            checksums['checksums'][metadata.model_id] = {
                'algorithm': metadata.algorithm,
                'value': metadata.checksum
            }
            checksums['last_updated'] = datetime.utcnow().isoformat()
            self._save_checksums(checksums)
            
            self.audit.log_event(
                event_type='MODEL_REGISTRY',
                action='REGISTER',
                resource=model_path,
                status='SUCCESS',
                details={'model_id': metadata.model_id}
            )
            
        except Exception as e:
            self.audit.log_event(
                event_type='MODEL_REGISTRY',
                action='REGISTER',
                resource=model_path,
                status='FAILED',
                details={'error': str(e)}
            )
            raise
            
    def verify_model(self, model_path: str, model_id: str) -> bool:
        """Verify model integrity against registered checksum."""
        return self.security.verify_model_integrity(
            model_path,
            self.checksums_file
        )
        
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a registered model."""
        try:
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            metadata_dict = all_metadata.get(model_id)
            if metadata_dict:
                return ModelMetadata(**metadata_dict)
            return None
            
        except Exception as e:
            self.audit.log_event(
                event_type='MODEL_REGISTRY',
                action='GET_METADATA',
                resource=model_id,
                status='FAILED',
                details={'error': str(e)}
            )
            return None
            
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        try:
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            return list(all_metadata.keys())
        except Exception:
            return []
