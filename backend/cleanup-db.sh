#!/bin/bash

# =============================================================================
# H2 DATABASE CLEANUP SCRIPT
# =============================================================================
# Cleans up H2 database files and processes to prevent locking issues

echo "🧹 Cleaning up H2 database files and processes..."

# Kill any running Spring Boot processes
echo "Stopping any running Spring Boot processes..."
pkill -f "spring-boot" 2>/dev/null || true
pkill -f "smartshoe" 2>/dev/null || true
pkill -f "api-3.0.0.jar" 2>/dev/null || true
pkill -f "java.*smartshoe" 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 3

# Remove H2 database files
echo "Removing H2 database files..."
find . -name "*.mv.db" -type f -delete 2>/dev/null || true
find . -name "*.trace.db" -type f -delete 2>/dev/null || true
find . -name "*.lock.db" -type f -delete 2>/dev/null || true

# Remove data directory if empty
if [ -d "data" ] && [ -z "$(ls -A data 2>/dev/null)" ]; then
    rmdir data 2>/dev/null || true
fi

# Kill any H2 server processes on port 9090
lsof -ti:9090 | xargs kill -9 2>/dev/null || true

echo "✅ Cleanup completed!"
echo ""
echo "Fixed H2 configuration issues:"
echo "  ❌ Removed incompatible DB_CLOSE_ON_EXIT=FALSE with AUTO_SERVER"
echo "  ✅ Using AUTO_SERVER=TRUE on port 9090"
echo "  ✅ Test profile uses in-memory database"
echo ""
echo "You can now run:"
echo "  mvn test          - Run tests with in-memory database"
echo "  mvn spring-boot:run - Start application with file database"