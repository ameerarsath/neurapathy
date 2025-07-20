#!/bin/bash

# =============================================================================
# SMART SHOE FRONTEND DEPLOYMENT SCRIPT FOR EC2
# =============================================================================
# Deploy React frontend to EC2 (to be served by Nginx)
# Usage: ./deploy-frontend-to-ec2.sh [EC2_IP] [PEM_FILE]

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

EC2_IP="${1:-13.201.120.175}"
PEM_FILE="${2:-~/.ssh/your-key.pem}"
EC2_USER="ec2-user"
FRONTEND_DIR="../frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# MAIN DEPLOYMENT
# =============================================================================

main() {
    echo "======================================================================="
    echo "                   SMART SHOE FRONTEND DEPLOYMENT"
    echo "======================================================================="
    
    log_info "Building React frontend..."
    
    # Check if frontend directory exists
    if [ ! -d "$FRONTEND_DIR" ]; then
        log_error "Frontend directory not found: $FRONTEND_DIR"
        exit 1
    fi
    
    # Build frontend
    cd "$FRONTEND_DIR"
    
    # Install dependencies if node_modules doesn't exist
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install
    fi
    
    # Configure API endpoints for production
    log_info "Configuring API endpoints for EC2 IP: $EC2_IP"
    
    # Update API configuration if .env.production exists
    if [ -f ".env.production" ]; then
        # Update existing production config
        sed -i.bak "s|REACT_APP_API_URL=.*|REACT_APP_API_URL=http://$EC2_IP/api|g" .env.production
        sed -i.bak "s|REACT_APP_ML_API_URL=.*|REACT_APP_ML_API_URL=http://$EC2_IP/api/ml|g" .env.production
    else
        # Create production config
        cat > .env.production << EOF
REACT_APP_API_URL=http://$EC2_IP/api
REACT_APP_ML_API_URL=http://$EC2_IP/api/ml
REACT_APP_ENV=production
EOF
    fi
    
    # Build for production
    log_info "Building for production..."
    npm run build
    
    if [ ! -d "dist" ]; then
        log_error "Build failed - dist directory not found"
        exit 1
    fi
    
    # Deploy to EC2
    log_info "Deploying to EC2: $EC2_IP"
    
    # Create web directory
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        sudo mkdir -p /var/www/html
        sudo chown -R $EC2_USER:$EC2_USER /var/www/html
    "
    
    # Upload frontend files
    log_info "Uploading frontend files..."
    rsync -avz --delete -e "ssh -i $PEM_FILE" dist/ "$EC2_USER@$EC2_IP:/var/www/html/"
    
    # Set proper permissions
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "
        sudo chown -R nginx:nginx /var/www/html 2>/dev/null || sudo chown -R apache:apache /var/www/html 2>/dev/null || true
        sudo chmod -R 755 /var/www/html
    "
    
    log_success "Frontend deployed successfully!"
    
    echo ""
    echo "🌐 Frontend is now available at: http://$EC2_IP"
    echo ""
    
    cd - > /dev/null
}

main "$@"