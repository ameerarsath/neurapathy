"""
Enhanced model security with audit logging.
"""
import os
import json
import logging
import hashlib
from typing import Dict, Optional
from cryptography.fernet import Fernet
import numpy as np

from .audit import SecureAuditLogger

class ModelSecurity:
    """Handles model encryption and integrity verification."""
    
    def __init__(self, key_path: Optional[str] = None, log_dir: Optional[str] = None):
        """Initialize model security with optional encryption key."""
        self.key_path = key_path or os.path.join(
            os.path.dirname(__file__), 'model_key.key'
        )
        self._init_encryption()
        
        # Initialize audit logger
        log_dir = log_dir or os.path.join(os.path.dirname(__file__), 'logs')
        self.audit = SecureAuditLogger(log_dir)
        
    def _init_encryption(self) -> None:
        """Initialize or load encryption key."""
        try:
            if os.path.exists(self.key_path):
                with open(self.key_path, 'rb') as f:
                    self.key = f.read()
            else:
                self.key = Fernet.generate_key()
                os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
                with open(self.key_path, 'wb') as f:
                    f.write(self.key)
                    
            self.cipher = Fernet(self.key)
            self.audit.log_event(
                event_type='SECURITY_INIT',
                action='INIT_ENCRYPTION',
                resource='encryption_key',
                status='SUCCESS'
            )
        except Exception as e:
            self.audit.log_event(
                event_type='SECURITY_INIT',
                action='INIT_ENCRYPTION',
                resource='encryption_key',
                status='FAILED',
                details={'error': str(e)}
            )
            raise
            
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt binary data."""
        try:
            encrypted = self.cipher.encrypt(data)
            self.audit.log_event(
                event_type='ENCRYPTION',
                action='ENCRYPT',
                resource='model_data',
                status='SUCCESS',
                details={'size': len(data)}
            )
            return encrypted
        except Exception as e:
            self.audit.log_event(
                event_type='ENCRYPTION',
                action='ENCRYPT',
                resource='model_data',
                status='FAILED',
                details={'error': str(e)}
            )
            raise
            
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt binary data."""
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            self.audit.log_event(
                event_type='ENCRYPTION',
                action='DECRYPT',
                resource='model_data',
                status='SUCCESS',
                details={'size': len(encrypted_data)}
            )
            return decrypted
        except Exception as e:
            self.audit.log_event(
                event_type='ENCRYPTION',
                action='DECRYPT',
                resource='model_data',
                status='FAILED',
                details={'error': str(e)}
            )
            raise
            
    def verify_model_integrity(self, model_path: str, checksums_path: str) -> bool:
        """Verify model file integrity using stored checksums."""
        try:
            if not os.path.exists(model_path):
                self.audit.log_event(
                    event_type='INTEGRITY_CHECK',
                    action='VERIFY',
                    resource=model_path,
                    status='FAILED',
                    details={'error': 'Model file not found'}
                )
                return False
                
            # Load checksums
            with open(checksums_path, 'r') as f:
                checksums = json.load(f)
                
            # Calculate current checksum
            with open(model_path, 'rb') as f:
                file_bytes = f.read()
                current_hash = hashlib.sha256(file_bytes).hexdigest()
                
            # Get model type from filename
            model_type = os.path.basename(model_path).split('_')[0]
            expected_hash = checksums['checksums'].get(
                f"{model_type}_model", {}
            ).get('value')
            
            if not expected_hash:
                self.audit.log_event(
                    event_type='INTEGRITY_CHECK',
                    action='VERIFY',
                    resource=model_path,
                    status='FAILED',
                    details={'error': f'No checksum found for {model_type}'}
                )
                return False
                
            is_valid = current_hash == expected_hash
            self.audit.log_event(
                event_type='INTEGRITY_CHECK',
                action='VERIFY',
                resource=model_path,
                status='SUCCESS' if is_valid else 'FAILED',
                details={
                    'expected_hash': expected_hash,
                    'current_hash': current_hash
                }
            )
            return is_valid
            
        except Exception as e:
            self.audit.log_event(
                event_type='INTEGRITY_CHECK',
                action='VERIFY',
                resource=model_path,
                status='ERROR',
                details={'error': str(e)}
            )
            return False
            
    def encrypt_predictions(self, predictions: np.ndarray) -> bytes:
        """Encrypt sensitive prediction data."""
        return self.encrypt_data(predictions.tobytes())
        
    def decrypt_predictions(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt predictions back to numpy array."""
        decrypted = self.decrypt_data(encrypted_data)
        return np.frombuffer(decrypted)
