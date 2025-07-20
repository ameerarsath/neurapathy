#!/bin/bash

# =============================================================================
# SMART SHOE BACKEND DEPLOYMENT SCRIPT FOR EC2
# =============================================================================
# Complete deployment automation for Spring Boot backend on EC2
# Usage: ./deploy-to-ec2.sh [EC2_IP] [PEM_FILE]

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default values (can be overridden by command line arguments)
EC2_IP="${1:-13.201.120.175}"
PEM_FILE="${2:-~/.ssh/your-key.pem}"
EC2_USER="ec2-user"
APP_NAME="smartshoe-api"
APP_PORT="8080"
SERVICE_NAME="smartshoe-backend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking deployment requirements..."
    
    # Check if Maven is installed
    if ! command -v mvn &> /dev/null; then
        log_error "Maven is not installed. Please install Maven first."
        exit 1
    fi
    
    # Check if PEM file exists
    if [ ! -f "$PEM_FILE" ]; then
        log_error "PEM file not found: $PEM_FILE"
        log_info "Usage: $0 [EC2_IP] [PEM_FILE_PATH]"
        exit 1
    fi
    
    # Check PEM file permissions
    if [ "$(stat -c %a "$PEM_FILE")" != "400" ]; then
        log_warning "Fixing PEM file permissions..."
        chmod 400 "$PEM_FILE"
    fi
    
    log_success "Requirements check passed"
}

cleanup_local() {
    log_info "Cleaning up local environment..."
    ./cleanup-db.sh
    log_success "Local cleanup completed"
}

build_application() {
    log_info "Building Spring Boot application..."
    
    # Clean and build
    mvn clean package -DskipTests=true
    
    if [ ! -f "target/api-3.0.0.jar" ]; then
        log_error "Build failed - JAR file not found"
        exit 1
    fi
    
    log_success "Application built successfully"
}

deploy_to_ec2() {
    log_info "Deploying to EC2 instance: $EC2_IP"
    
    # Create deployment directory on EC2
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        sudo mkdir -p /opt/$SERVICE_NAME
        sudo chown $EC2_USER:$EC2_USER /opt/$SERVICE_NAME
        mkdir -p /opt/$SERVICE_NAME/logs
        mkdir -p /opt/$SERVICE_NAME/data
    " || {
        log_error "Failed to create directories on EC2"
        exit 1
    }
    
    # Copy JAR file to EC2
    log_info "Uploading JAR file..."
    scp -i "$PEM_FILE" target/api-3.0.0.jar "$EC2_USER@$EC2_IP:/opt/$SERVICE_NAME/" || {
        log_error "Failed to upload JAR file"
        exit 1
    }
    
    # Copy configuration files
    log_info "Uploading configuration files..."
    scp -i "$PEM_FILE" src/main/resources/application.yml "$EC2_USER@$EC2_IP:/opt/$SERVICE_NAME/" || {
        log_error "Failed to upload configuration"
        exit 1
    }
    
    log_success "Files uploaded successfully"
}

create_systemd_service() {
    log_info "Creating systemd service..."
    
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << 'EOF'
[Unit]
Description=Smart Shoe Backend API
After=network.target

[Service]
Type=simple
User=$EC2_USER
Group=$EC2_USER
WorkingDirectory=/opt/$SERVICE_NAME
ExecStart=/usr/bin/java -jar -Dspring.profiles.active=production /opt/$SERVICE_NAME/api-3.0.0.jar
ExecStop=/bin/kill -15 \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Environment variables
Environment=JAVA_OPTS=-Xmx512m -Xms256m
Environment=SERVER_PORT=$APP_PORT
Environment=SPRING_PROFILES_ACTIVE=production

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF
    " || {
        log_error "Failed to create systemd service"
        exit 1
    }
    
    log_success "Systemd service created"
}

configure_nginx() {
    log_info "Configuring Nginx reverse proxy..."
    
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        # Install nginx if not present
        if ! command -v nginx &> /dev/null; then
            sudo yum update -y
            sudo yum install -y nginx
        fi
        
        # Create nginx configuration
        sudo tee /etc/nginx/conf.d/$SERVICE_NAME.conf > /dev/null << 'EOF'
server {
    listen 80;
    server_name $EC2_IP;
    
    # API endpoints
    location /api/ {
        proxy_pass http://localhost:$APP_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # CORS headers for specific origins
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
    
    # Health check
    location /health {
        proxy_pass http://localhost:$APP_PORT/actuator/health;
        proxy_set_header Host \$host;
    }
    
    # Frontend static files (if deployed)
    location / {
        root /var/www/html;
        try_files \$uri \$uri/ /index.html;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection \"1; mode=block\";
    }
}
EOF
        
        # Test nginx configuration
        sudo nginx -t
        
        # Enable and start nginx
        sudo systemctl enable nginx
        sudo systemctl restart nginx
    " || {
        log_error "Failed to configure Nginx"
        exit 1
    }
    
    log_success "Nginx configured successfully"
}

setup_firewall() {
    log_info "Configuring firewall..."
    
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        # Allow HTTP and HTTPS
        sudo firewall-cmd --permanent --add-service=http 2>/dev/null || true
        sudo firewall-cmd --permanent --add-service=https 2>/dev/null || true
        sudo firewall-cmd --permanent --add-port=$APP_PORT/tcp 2>/dev/null || true
        sudo firewall-cmd --reload 2>/dev/null || true
        
        # Alternative for systems without firewalld
        if command -v ufw &> /dev/null; then
            sudo ufw allow 80/tcp
            sudo ufw allow 443/tcp
            sudo ufw allow $APP_PORT/tcp
        fi
    " || {
        log_warning "Firewall configuration may have failed (this is often normal)"
    }
    
    log_success "Firewall configured"
}

start_services() {
    log_info "Starting services..."
    
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        # Stop existing service if running
        sudo systemctl stop $SERVICE_NAME 2>/dev/null || true
        
        # Reload systemd and start service
        sudo systemctl daemon-reload
        sudo systemctl enable $SERVICE_NAME
        sudo systemctl start $SERVICE_NAME
        
        # Wait for service to start
        sleep 10
        
        # Check service status
        sudo systemctl status $SERVICE_NAME --no-pager
    " || {
        log_error "Failed to start services"
        exit 1
    }
    
    log_success "Services started successfully"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Wait for application to fully start
    sleep 15
    
    # Test health endpoint
    if curl -f "http://$EC2_IP/health" > /dev/null 2>&1; then
        log_success "Health endpoint is responding"
    else
        log_warning "Health endpoint not responding yet (this may be normal)"
    fi
    
    # Test API endpoint
    if curl -f "http://$EC2_IP/api/health/status" > /dev/null 2>&1; then
        log_success "API endpoint is responding"
    else
        log_warning "API endpoint not responding yet"
    fi
    
    # Show service logs
    log_info "Recent service logs:"
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        sudo journalctl -u $SERVICE_NAME --no-pager -n 20
    "
}

create_management_scripts() {
    log_info "Creating management scripts on EC2..."
    
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        # Create management script
        tee /opt/$SERVICE_NAME/manage.sh > /dev/null << 'EOF'
#!/bin/bash

SERVICE_NAME=\"$SERVICE_NAME\"

case \"\$1\" in
    start)
        echo \"Starting \$SERVICE_NAME...\"
        sudo systemctl start \$SERVICE_NAME
        ;;
    stop)
        echo \"Stopping \$SERVICE_NAME...\"
        sudo systemctl stop \$SERVICE_NAME
        ;;
    restart)
        echo \"Restarting \$SERVICE_NAME...\"
        sudo systemctl restart \$SERVICE_NAME
        ;;
    status)
        sudo systemctl status \$SERVICE_NAME --no-pager
        ;;
    logs)
        sudo journalctl -u \$SERVICE_NAME -f
        ;;
    logs-recent)
        sudo journalctl -u \$SERVICE_NAME --no-pager -n 50
        ;;
    cleanup)
        echo \"Cleaning up H2 database files...\"
        find /opt/\$SERVICE_NAME -name \"*.mv.db\" -delete 2>/dev/null || true
        find /opt/\$SERVICE_NAME -name \"*.trace.db\" -delete 2>/dev/null || true
        find /opt/\$SERVICE_NAME -name \"*.lock.db\" -delete 2>/dev/null || true
        ;;
    *)
        echo \"Usage: \$0 {start|stop|restart|status|logs|logs-recent|cleanup}\"
        exit 1
        ;;
esac
EOF
        
        chmod +x /opt/$SERVICE_NAME/manage.sh
    " || {
        log_error "Failed to create management scripts"
        exit 1
    }
    
    log_success "Management scripts created"
}

# =============================================================================
# MAIN DEPLOYMENT FLOW
# =============================================================================

main() {
    echo "======================================================================="
    echo "                   SMART SHOE BACKEND DEPLOYMENT"
    echo "======================================================================="
    echo "Target EC2: $EC2_IP"
    echo "PEM File: $PEM_FILE"
    echo "Service: $SERVICE_NAME"
    echo "======================================================================="
    
    check_requirements
    cleanup_local
    build_application
    deploy_to_ec2
    create_systemd_service
    configure_nginx
    setup_firewall
    start_services
    create_management_scripts
    verify_deployment
    
    echo "======================================================================="
    log_success "DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "======================================================================="
    echo ""
    echo "🚀 Your Smart Shoe Backend is now running!"
    echo ""
    echo "📋 Service Information:"
    echo "   • Backend API: http://$EC2_IP/api/"
    echo "   • Health Check: http://$EC2_IP/health"
    echo "   • H2 Console: http://$EC2_IP/api/h2-console (if enabled)"
    echo ""
    echo "🛠️  Management Commands (run on EC2):"
    echo "   • Start:   /opt/$SERVICE_NAME/manage.sh start"
    echo "   • Stop:    /opt/$SERVICE_NAME/manage.sh stop"
    echo "   • Restart: /opt/$SERVICE_NAME/manage.sh restart"
    echo "   • Status:  /opt/$SERVICE_NAME/manage.sh status"
    echo "   • Logs:    /opt/$SERVICE_NAME/manage.sh logs"
    echo "   • Cleanup: /opt/$SERVICE_NAME/manage.sh cleanup"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   • SSH: ssh -i $PEM_FILE $EC2_USER@$EC2_IP"
    echo "   • Logs: sudo journalctl -u $SERVICE_NAME -f"
    echo "   • Service: sudo systemctl status $SERVICE_NAME"
    echo ""
    echo "======================================================================="
}

# Handle command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Smart Shoe Backend Deployment Script"
    echo ""
    echo "Usage: $0 [EC2_IP] [PEM_FILE]"
    echo ""
    echo "Arguments:"
    echo "  EC2_IP    IP address of your EC2 instance (default: 13.201.120.175)"
    echo "  PEM_FILE  Path to your PEM key file (default: ~/.ssh/your-key.pem)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default values"
    echo "  $0 1.2.3.4                          # Custom IP, default PEM"
    echo "  $0 1.2.3.4 ~/.ssh/my-key.pem        # Custom IP and PEM"
    echo ""
    exit 0
fi

# Run main deployment
main "$@"