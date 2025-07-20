#!/bin/bash

# =============================================================================
# CORS CONFIGURATION SCRIPT FOR EC2
# =============================================================================
# Configures CORS for both Spring Boot backend and Nginx frontend
# Usage: ./configure-cors.sh [EC2_IP] [PEM_FILE]

set -e

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
    echo -e "${BLUE}[CORS-INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[CORS-SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[CORS-WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[CORS-ERROR]${NC} $1"
}

configure_nginx_cors() {
    log_info "Configuring Nginx CORS for IP: $EC2_IP"
    
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        # Backup existing configuration
        sudo cp /etc/nginx/conf.d/smartshoe-backend.conf /etc/nginx/conf.d/smartshoe-backend.conf.backup 2>/dev/null || true
        
        # Create updated Nginx configuration with proper CORS
        sudo tee /etc/nginx/conf.d/smartshoe-backend.conf > /dev/null << 'EOF'
server {
    listen 80;
    server_name $EC2_IP;
    
    # API endpoints with CORS
    location /api/ {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # CORS headers for specific origin
        add_header 'Access-Control-Allow-Origin' 'http://$EC2_IP' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,Accept,Origin,Access-Control-Request-Method,Access-Control-Request-Headers' always;
        add_header 'Access-Control-Allow-Credentials' 'true' always;
        add_header 'Access-Control-Expose-Headers' 'Authorization,Content-Disposition' always;
        
        # Handle preflight OPTIONS requests
        if (\$request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' 'http://$EC2_IP' always;
            add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD' always;
            add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,Accept,Origin,Access-Control-Request-Method,Access-Control-Request-Headers' always;
            add_header 'Access-Control-Allow-Credentials' 'true' always;
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }
    }
    
    # Health check with CORS
    location /health {
        proxy_pass http://localhost:8080/actuator/health;
        proxy_set_header Host \$host;
        
        # CORS for health endpoint
        add_header 'Access-Control-Allow-Origin' 'http://$EC2_IP' always;
        add_header 'Access-Control-Allow-Methods' 'GET, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
    }
    
    # Frontend static files with CORS
    location / {
        root /var/www/html;
        try_files \$uri \$uri/ /index.html;
        
        # CORS for frontend assets
        add_header 'Access-Control-Allow-Origin' 'http://$EC2_IP' always;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection \"1; mode=block\";
    }
}
EOF
        
        # Test nginx configuration
        sudo nginx -t
        
        # Reload nginx
        sudo systemctl reload nginx
    " || {
        log_error "Failed to configure Nginx CORS"
        exit 1
    }
    
    log_success "Nginx CORS configured for $EC2_IP"
}

test_cors_configuration() {
    log_info "Testing CORS configuration..."
    
    # Test preflight request
    log_info "Testing preflight OPTIONS request..."
    PREFLIGHT_RESULT=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Origin: http://$EC2_IP" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Content-Type,Authorization" \
        -X OPTIONS \
        "http://$EC2_IP/api/health/status" 2>/dev/null || echo "000")
    
    if [ "$PREFLIGHT_RESULT" = "204" ] || [ "$PREFLIGHT_RESULT" = "200" ]; then
        log_success "Preflight request successful (HTTP $PREFLIGHT_RESULT)"
    else
        log_warning "Preflight request returned HTTP $PREFLIGHT_RESULT"
    fi
    
    # Test actual API request
    log_info "Testing actual API request..."
    API_RESULT=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Origin: http://$EC2_IP" \
        "http://$EC2_IP/api/health/status" 2>/dev/null || echo "000")
    
    if [ "$API_RESULT" = "200" ]; then
        log_success "API request successful (HTTP $API_RESULT)"
    else
        log_warning "API request returned HTTP $API_RESULT"
    fi
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    HEALTH_RESULT=$(curl -s -o /dev/null -w "%{http_code}" \
        "http://$EC2_IP/health" 2>/dev/null || echo "000")
    
    if [ "$HEALTH_RESULT" = "200" ]; then
        log_success "Health endpoint successful (HTTP $HEALTH_RESULT)"
    else
        log_warning "Health endpoint returned HTTP $HEALTH_RESULT"
    fi
}

show_cors_headers() {
    log_info "Checking CORS headers from server..."
    
    echo ""
    echo "🔍 CORS Headers for preflight request:"
    curl -s -I \
        -H "Origin: http://$EC2_IP" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Content-Type,Authorization" \
        -X OPTIONS \
        "http://$EC2_IP/api/health/status" | grep -i "access-control" || echo "No CORS headers found"
    
    echo ""
    echo "🔍 CORS Headers for actual request:"
    curl -s -I \
        -H "Origin: http://$EC2_IP" \
        "http://$EC2_IP/api/health/status" | grep -i "access-control" || echo "No CORS headers found"
}

create_frontend_cors_config() {
    log_info "Creating frontend CORS configuration..."
    
    # Create a simple frontend test page
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        sudo mkdir -p /var/www/html
        sudo tee /var/www/html/cors-test.html > /dev/null << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>CORS Test - Smart Shoe</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .result { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
        .info { background-color: #d1ecf1; color: #0c5460; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Smart Shoe CORS Test</h1>
    <p>This page tests CORS connectivity to the backend API.</p>
    
    <button onclick=\"testHealthEndpoint()\">Test Health Endpoint</button>
    <button onclick=\"testAPIEndpoint()\">Test API Endpoint</button>
    <button onclick=\"testCredentialsEndpoint()\">Test Credentials Endpoint</button>
    
    <div id=\"results\"></div>
    
    <script>
        const API_BASE = 'http://$EC2_IP/api';
        const HEALTH_URL = 'http://$EC2_IP/health';
        
        function addResult(message, type = 'info') {
            const results = document.getElementById('results');
            const div = document.createElement('div');
            div.className = 'result ' + type;
            div.innerHTML = new Date().toLocaleTimeString() + ': ' + message;
            results.appendChild(div);
        }
        
        async function testHealthEndpoint() {
            try {
                const response = await fetch(HEALTH_URL);
                const data = await response.text();
                addResult('Health endpoint: ' + response.status + ' - ' + data.substring(0, 100), response.ok ? 'success' : 'error');
            } catch (error) {
                addResult('Health endpoint error: ' + error.message, 'error');
            }
        }
        
        async function testAPIEndpoint() {
            try {
                const response = await fetch(API_BASE + '/health/status');
                const data = await response.text();
                addResult('API endpoint: ' + response.status + ' - ' + data, response.ok ? 'success' : 'error');
            } catch (error) {
                addResult('API endpoint error: ' + error.message, 'error');
            }
        }
        
        async function testCredentialsEndpoint() {
            try {
                const response = await fetch(API_BASE + '/credentials/test');
                const data = await response.text();
                addResult('Credentials endpoint: ' + response.status + ' - ' + data, response.ok ? 'success' : 'error');
            } catch (error) {
                addResult('Credentials endpoint error: ' + error.message, 'error');
            }
        }
        
        // Auto-test on page load
        window.onload = function() {
            addResult('Page loaded. Testing CORS configuration...', 'info');
            setTimeout(testHealthEndpoint, 1000);
            setTimeout(testAPIEndpoint, 2000);
            setTimeout(testCredentialsEndpoint, 3000);
        };
    </script>
</body>
</html>
EOF
        
        sudo chown -R nginx:nginx /var/www/html 2>/dev/null || sudo chown -R apache:apache /var/www/html 2>/dev/null || true
        sudo chmod -R 755 /var/www/html
    " || {
        log_error "Failed to create frontend CORS test page"
        exit 1
    }
    
    log_success "Frontend CORS test page created at: http://$EC2_IP/cors-test.html"
}

main() {
    echo "======================================================================="
    echo "                      CORS CONFIGURATION TOOL"
    echo "======================================================================="
    echo "Target EC2: $EC2_IP"
    echo "PEM File: $PEM_FILE"
    echo "======================================================================="
    
    configure_nginx_cors
    create_frontend_cors_config
    
    # Wait a moment for changes to take effect
    sleep 3
    
    test_cors_configuration
    show_cors_headers
    
    echo ""
    echo "======================================================================="
    log_success "CORS CONFIGURATION COMPLETED!"
    echo "======================================================================="
    echo ""
    echo "🌐 Test your CORS configuration:"
    echo "   • CORS Test Page: http://$EC2_IP/cors-test.html"
    echo "   • API Health: http://$EC2_IP/api/health/status"
    echo "   • System Health: http://$EC2_IP/health"
    echo ""
    echo "🔧 CORS is configured for:"
    echo "   • Frontend Origin: http://$EC2_IP"
    echo "   • Backend API: http://$EC2_IP/api/"
    echo "   • All HTTP methods: GET, POST, PUT, PATCH, DELETE, OPTIONS"
    echo "   • Credentials: Enabled"
    echo ""
    echo "======================================================================="
}

# Handle help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "CORS Configuration Script"
    echo ""
    echo "Usage: $0 [EC2_IP] [PEM_FILE]"
    echo ""
    echo "This script configures CORS for both frontend and backend communication"
    echo "on your EC2 instance with IP: $EC2_IP"
    echo ""
    exit 0
fi

main "$@"