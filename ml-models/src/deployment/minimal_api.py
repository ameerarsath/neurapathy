"""
Minimal ML API for Smart Shoe Application
This is a simplified version that works with basic Python libraries only
"""

import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import datetime

class MLAPIHandler(BaseHTTPRequestHandler):
    def check_auth(self):
        """Simple token authentication"""
        auth_header = self.headers.get('Authorization')
        if not auth_header:
            return False
        
        # Check for Bearer token
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            # Accept the development token
            return token == 'ml_api_dev_token'
        
        return False
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Health check endpoint
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "status": "healthy",
                "timestamp": datetime.datetime.now().isoformat(),
                "service": "ML API",
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(response).encode())
        # Models endpoint
        elif parsed_path.path == '/models':
            if not self.check_auth():
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
                return
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "available_models": [
                    "neuropathy_progression",
                    "glucose_complications", 
                    "anomaly_detection",
                    "risk_stratification"
                ],
                "model_count": 4,
                "status": "ready",
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        # Check authentication for prediction endpoints
        if not self.check_auth():
            self.send_response(401)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
            return
        
        # Handle CORS preflight
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        
        # Read request body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            request_data = json.loads(post_data.decode('utf-8'))
        except:
            self.send_error(400, "Invalid JSON")
            return
        
        # Route to appropriate prediction endpoint
        if parsed_path.path == '/predict/neuropathy-progression':
            response = self.predict_neuropathy(request_data)
        elif parsed_path.path == '/predict/glucose-complications':
            response = self.predict_glucose(request_data)
        elif parsed_path.path == '/predict/anomaly-detection':
            response = self.predict_anomaly(request_data)
        elif parsed_path.path == '/predict/risk-stratification':
            response = self.predict_risk(request_data)
        else:
            self.send_error(404)
            return
        
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def predict_neuropathy(self, data):
        """Simulate neuropathy progression prediction"""
        # Generate realistic prediction based on patient data
        risk_score = random.uniform(0.1, 0.9)
        confidence = random.uniform(0.7, 0.95)
        
        # Adjust based on age (older patients higher risk)
        features = data.get('features', {})
        age = features.get('age', 50)
        if age > 60:
            risk_score += 0.1
        
        risk_level = "LOW" if risk_score < 0.4 else "MEDIUM" if risk_score < 0.7 else "HIGH"
        
        return {
            "prediction": risk_score,
            "confidence": confidence,
            "risk_level": risk_level,
            "model_type": "neuropathy_progression",
            "timestamp": datetime.datetime.now().isoformat(),
            "additional_data": {
                "progression_rate": "moderate" if risk_score > 0.5 else "slow",
                "recommended_monitoring": "monthly" if risk_score > 0.6 else "quarterly"
            }
        }
    
    def predict_glucose(self, data):
        """Simulate glucose complications prediction"""
        risk_score = random.uniform(0.2, 0.8)
        confidence = random.uniform(0.75, 0.92)
        
        risk_level = "LOW" if risk_score < 0.4 else "MEDIUM" if risk_score < 0.7 else "HIGH"
        
        return {
            "prediction": risk_score,
            "confidence": confidence,
            "risk_level": risk_level,
            "model_type": "glucose_complications",
            "timestamp": datetime.datetime.now().isoformat(),
            "additional_data": {
                "complication_type": "vascular" if risk_score > 0.6 else "metabolic",
                "intervention_needed": risk_score > 0.7
            }
        }
    
    def predict_anomaly(self, data):
        """Simulate anomaly detection"""
        anomaly_score = random.uniform(0.0, 0.3)  # Most readings should be normal
        confidence = random.uniform(0.8, 0.98)
        
        is_anomaly = anomaly_score > 0.2
        risk_level = "HIGH" if is_anomaly else "LOW"
        
        return {
            "prediction": anomaly_score,
            "confidence": confidence,
            "risk_level": risk_level,
            "model_type": "anomaly_detection",
            "timestamp": datetime.datetime.now().isoformat(),
            "additional_data": {
                "anomaly_detected": is_anomaly,
                "anomaly_type": "sensor_drift" if is_anomaly else "normal",
                "requires_recalibration": anomaly_score > 0.25
            }
        }
    
    def predict_risk(self, data):
        """Simulate overall risk stratification"""
        risk_score = random.uniform(0.1, 0.85)
        confidence = random.uniform(0.78, 0.94)
        
        risk_level = "LOW" if risk_score < 0.4 else "MEDIUM" if risk_score < 0.7 else "HIGH"
        
        return {
            "prediction": risk_score,
            "confidence": confidence,
            "risk_level": risk_level,
            "model_type": "risk_stratification",
            "timestamp": datetime.datetime.now().isoformat(),
            "additional_data": {
                "risk_category": "cardiovascular" if risk_score > 0.6 else "metabolic",
                "follow_up_required": risk_score > 0.5,
                "urgency": "high" if risk_score > 0.8 else "moderate" if risk_score > 0.5 else "low"
            }
        }

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, MLAPIHandler)
    print(f"🚀 ML API Server starting on http://localhost:{port}")
    print("📍 Available endpoints:")
    print("  GET  /health")
    print("  POST /predict/neuropathy-progression")
    print("  POST /predict/glucose-complications") 
    print("  POST /predict/anomaly-detection")
    print("  POST /predict/risk-stratification")
    print("\n✅ Server ready for requests...")
    httpd.serve_forever()

if __name__ == '__main__':
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")