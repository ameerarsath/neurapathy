#!/bin/bash

# =============================================================================
# SMART SHOE PRODUCTION BUILD SCRIPT
# =============================================================================
# Builds the Smart Shoe frontend for production deployment to AWS EC2

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

echo -e "${BLUE}🚀 Building Smart Shoe for Production Deployment${NC}"
echo "================================================="
echo ""

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ Error: Please run this script from the smartshoe-frontend directory${NC}"
    exit 1
fi

# Display production configuration
echo -e "${BLUE}📋 Production Configuration:${NC}"
echo "   Server: 13.201.120.175:8080"
echo "   API URL: http://13.201.120.175:8080/api"
echo "   ML API: http://13.201.120.175:8080/api/ml"
echo "   Environment: production"
echo ""

# Install dependencies
print_info "Installing dependencies..."
npm ci --legacy-peer-deps

# Validate environment variables
print_info "Validating environment configuration..."
if [ ! -f ".env.production" ]; then
    print_warning ".env.production file not found, using defaults"
fi

# Clean previous builds
print_info "Cleaning previous builds..."
rm -rf dist/
rm -rf android/app/build/
rm -rf ios/App/build/

# Build web assets for production
print_info "Building web assets for production..."
NODE_ENV=production npm run build

# Check if build was successful
if [ ! -d "dist" ]; then
    echo -e "${RED}❌ Web build failed - dist directory not found${NC}"
    exit 1
fi

print_status "Web build completed successfully"

# Sync with Capacitor
print_info "Syncing with Capacitor..."
npx cap sync

# Copy additional mobile assets
print_info "Copying mobile assets..."
mkdir -p dist/icons
cp -r public/icons/* dist/icons/ 2>/dev/null || print_warning "No icons found in public/icons/"

# Generate build info
cat > dist/build-info.json << EOF
{
  "buildTime": "$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")",
  "version": "$(node -p "require('./package.json').version")",
  "environment": "production",
  "gitCommit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "gitBranch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "apiUrl": "http://13.201.120.175:8080",
  "nodeVersion": "$(node --version)",
  "buildHost": "$(hostname)"
}
EOF

print_status "Build info generated"

# Validate build output
print_info "Validating build output..."
BUILD_SIZE=$(du -sh dist/ | cut -f1)
echo "   Build size: $BUILD_SIZE"

# Count files
FILE_COUNT=$(find dist/ -type f | wc -l)
echo "   Total files: $FILE_COUNT"

# Check for critical files
CRITICAL_FILES=(
    "dist/index.html"
    "dist/manifest.json"
    "dist/assets"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -e "$file" ]; then
        echo -e "   ${GREEN}✓${NC} $file"
    else
        echo -e "   ${RED}✗${NC} $file (missing)"
    fi
done

echo ""
print_status "Production build completed successfully!"
echo ""

echo -e "${BLUE}📱 Next Steps for Mobile Deployment:${NC}"
echo ""
echo -e "${YELLOW}Android APK Build:${NC}"
echo "   1. npx cap open android"
echo "   2. In Android Studio: Build → Generate Signed Bundle/APK"
echo "   3. Choose APK or AAB format"
echo "   4. Sign with your release keystore"
echo ""
echo -e "${YELLOW}iOS IPA Build (macOS only):${NC}"
echo "   1. npx cap open ios"
echo "   2. In Xcode: Product → Archive"
echo "   3. Distribute App → App Store Connect or Enterprise"
echo "   4. Upload to App Store Connect"
echo ""
echo -e "${YELLOW}Web Deployment:${NC}"
echo "   1. Upload dist/ folder to your web server"
echo "   2. Configure nginx/apache to serve the SPA"
echo "   3. Set up SSL certificate"
echo "   4. Configure CORS on the backend"
echo ""

echo -e "${BLUE}🔗 Test URLs (after deployment):${NC}"
echo "   Web App: http://13.201.120.175:3000"
echo "   API Health: http://13.201.120.175:8080/actuator/health"
echo "   ML API Health: http://13.201.120.175:8080/api/ml/health"
echo ""

echo -e "${GREEN}🎉 Build process completed successfully!${NC}"
echo -e "${BLUE}📁 Output directory: ./dist/${NC}"