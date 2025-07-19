"""
Secure audit logging for model operations.
"""
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class AuditEvent:
    """Represents a security-relevant event."""
    timestamp: str
    event_type: str
    user_id: str
    action: str
    resource: str
    status: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary format."""
        return asdict(self)

class SecureAuditLogger:
    """Handles secure logging of model operations."""
    
    def __init__(self, log_dir: str):
        """Initialize secure audit logger."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure file handler
        self.audit_file = os.path.join(log_dir, 'model_audit.log')
        handler = logging.FileHandler(self.audit_file)
        handler.setLevel(logging.INFO)
        
        # Use JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        
        # Create logger
        self.logger = logging.getLogger('model_audit')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        
    def log_event(self, 
                  event_type: str,
                  action: str,
                  resource: str,
                  status: str,
                  user_id: str = 'system',
                  details: Optional[Dict] = None) -> None:
        """Log a security-relevant event."""
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            status=status,
            details=details
        )
        
        # Log as JSON for easier parsing
        self.logger.info(json.dumps(event.to_dict()))
        
    def log_model_access(self,
                        model_path: str,
                        action: str,
                        status: str,
                        details: Optional[Dict] = None) -> None:
        """Log model access events."""
        self.log_event(
            event_type='MODEL_ACCESS',
            action=action,
            resource=model_path,
            status=status,
            details=details
        )
        
    def log_validation_event(self,
                           model_type: str,
                           validation_type: str,
                           status: str,
                           details: Optional[Dict] = None) -> None:
        """Log model validation events."""
        self.log_event(
            event_type='MODEL_VALIDATION',
            action=validation_type,
            resource=model_type,
            status=status,
            details=details
        )
