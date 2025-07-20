#!/bin/bash

# =============================================================================
# FRONTEND DEPLOYMENT SCRIPT FOR EC2
# =============================================================================
# Builds and deploys React frontend to EC2 with proper API configuration
# Usage: ./deploy-frontend.sh [EC2_IP] [PEM_FILE]

set -e  # Exit on any error

# Configuration
EC2_IP="${1:-13.201.120.175}"
PEM_FILE="${2:-~/.ssh/your-key.pem}"
EC2_USER="ec2-user"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[FRONTEND-INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[FRONTEND-SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[FRONTEND-WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[FRONTEND-ERROR]${NC} $1"
}

echo "======================================================================="
echo "                   SMART SHOE FRONTEND DEPLOYMENT"
echo "======================================================================="
echo "Target EC2: $EC2_IP"
echo "PEM File: $PEM_FILE"
echo "======================================================================="

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    log_error "This script must be run from the frontend directory"
    log_info "Run: cd /mnt/d/project/diabetic-smart-shoe/frontend && ./deploy-frontend.sh"
    exit 1
fi

# Check requirements
log_info "Checking requirements..."
if ! command -v npm &> /dev/null; then
    log_error "npm is not installed"
    exit 1
fi

if [ ! -f "$PEM_FILE" ]; then
    log_error "PEM file not found: $PEM_FILE"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    log_info "Installing frontend dependencies..."
    npm install
fi

# Build for production
log_info "Building frontend for production..."
log_info "API URL: http://$EC2_IP/api"

# Create production environment file
cat > .env.production << EOF
VITE_API_BASE_URL=http://$EC2_IP/api
VITE_ML_API_BASE_URL=http://$EC2_IP/api/ml
VITE_WS_URL=ws://$EC2_IP/ws
VITE_ENV=production
VITE_VERSION=1.0.0
VITE_BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

# Build the frontend
npm run build:prod

# Verify build output
if [ ! -d "dist" ]; then
    log_error "Build failed - dist directory not found"
    exit 1
fi

log_success "Frontend built successfully"

# Deploy to EC2
log_info "Deploying to EC2: $EC2_IP"

# Create web directory on EC2
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
    sudo mkdir -p /var/www/html
    sudo chown -R $EC2_USER:$EC2_USER /var/www/html
"

# Upload frontend files
log_info "Uploading frontend files..."
rsync -avz --delete -e "ssh -i $PEM_FILE" dist/ "$EC2_USER@$EC2_IP:/var/www/html/"

# Set proper permissions
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
    # Set ownership for web server
    sudo chown -R nginx:nginx /var/www/html 2>/dev/null || sudo chown -R apache:apache /var/www/html 2>/dev/null || true
    sudo chmod -R 755 /var/www/html
    
    # Create a simple index.html if it doesn't exist
    if [ ! -f '/var/www/html/index.html' ]; then
        sudo tee /var/www/html/index.html > /dev/null << 'HTML'
<!DOCTYPE html>
<html>
<head>
    <title>Smart Shoe - Loading...</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .loading { color: #666; }
    </style>
</head>
<body>
    <h1>Smart Shoe Platform</h1>
    <p class=\"loading\">Loading application...</p>
    <script>
        // Redirect to the main app if it exists
        if (window.location.pathname === '/') {
            fetch('/app/index.html')
                .then(() => window.location.href = '/app/')
                .catch(() => console.log('Main app not found'));
        }
    </script>
</body>
</html>
HTML
    fi
"

# Create API connectivity test page
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
    cat > /var/www/html/api-test.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>API Connectivity Test</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }
        .test-result { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .endpoint { font-family: monospace; background: #f8f9fa; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>Smart Shoe API Connectivity Test</h1>
    <p>Testing connectivity to backend API at <span class=\"endpoint\">http://$EC2_IP/api</span></p>
    
    <div>
        <button onclick=\"testHealth()\">Test Health Endpoint</button>
        <button onclick=\"testAPI()\">Test API Status</button>
        <button onclick=\"testCredentials()\">Test Credentials</button>
        <button onclick=\"clearResults()\">Clear Results</button>
    </div>
    
    <div id=\"results\"></div>
    
    <script>
        const API_BASE = 'http://$EC2_IP/api';
        const HEALTH_URL = 'http://$EC2_IP/health';
        
        function addResult(message, type = 'info') {
            const results = document.getElementById('results');
            const div = document.createElement('div');
            div.className = 'test-result ' + type;
            div.innerHTML = new Date().toLocaleTimeString() + ': ' + message;
            results.appendChild(div);
            results.scrollTop = results.scrollHeight;
        }
        
        function clearResults() {
            document.getElementById('results').innerHTML = '';
        }
        
        async function testHealth() {
            addResult('Testing health endpoint...', 'info');
            try {
                const response = await fetch(HEALTH_URL);
                const data = await response.text();
                addResult('✅ Health endpoint: ' + response.status + ' - ' + data.substring(0, 100), 'success');
            } catch (error) {
                addResult('❌ Health endpoint error: ' + error.message, 'error');
            }
        }
        
        async function testAPI() {
            addResult('Testing API status endpoint...', 'info');
            try {
                const response = await fetch(API_BASE + '/health/status');
                const data = await response.text();
                addResult('✅ API status: ' + response.status + ' - ' + data, 'success');
            } catch (error) {
                addResult('❌ API error: ' + error.message, 'error');
            }
        }
        
        async function testCredentials() {
            addResult('Testing credentials endpoint...', 'info');
            try {
                const response = await fetch(API_BASE + '/credentials/test');
                const data = await response.text();
                addResult('✅ Credentials: ' + response.status + ' - ' + data, 'success');
            } catch (error) {
                addResult('❌ Credentials error: ' + error.message, 'error');
            }
        }
        
        // Auto-test on page load
        window.onload = function() {
            addResult('🚀 Page loaded. Auto-testing API connectivity...', 'info');
            setTimeout(testHealth, 1000);
            setTimeout(testAPI, 2000);
            setTimeout(testCredentials, 3000);
        };
    </script>
</body>
</html>
EOF
"

log_success "Frontend deployed successfully!"

echo ""
echo "======================================================================="
echo "                        DEPLOYMENT COMPLETED!"
echo "======================================================================="
echo ""
echo "🌐 Your frontend is now available at:"
echo "   • Main App: http://$EC2_IP/"
echo "   • API Test: http://$EC2_IP/api-test.html"
echo ""
echo "🔗 Backend endpoints:"
echo "   • API Base: http://$EC2_IP/api/"
echo "   • Health: http://$EC2_IP/health"
echo "   • ML API: http://$EC2_IP/api/ml/"
echo ""
echo "🧪 Test commands:"
echo "   curl http://$EC2_IP/health"
echo "   curl http://$EC2_IP/api/health/status"
echo ""
echo "🔧 Files deployed to: /var/www/html/"
echo "   • Frontend build files"
echo "   • API connectivity test page"
echo ""
echo "======================================================================="

# Final connectivity test
log_info "Running final connectivity test..."
if curl -f "http://$EC2_IP/" > /dev/null 2>&1; then
    log_success "✅ Frontend is accessible"
else
    log_warning "⚠️  Frontend may not be accessible yet (this can be normal)"
fi

if curl -f "http://$EC2_IP/health" > /dev/null 2>&1; then
    log_success "✅ Backend health endpoint is working"
else
    log_warning "⚠️  Backend may not be running"
fi

echo ""
log_success "🎉 Frontend deployment completed!"
echo "Visit http://$EC2_IP/api-test.html to test API connectivity"