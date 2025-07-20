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

# Wait a moment for processes to terminate
sleep 2

# Remove H2 database files
echo "Removing H2 database files..."
find . -name "*.mv.db" -type f -delete 2>/dev/null || true
find . -name "*.trace.db" -type f -delete 2>/dev/null || true
find . -name "*.lock.db" -type f -delete 2>/dev/null || true

# Remove data directory if empty
if [ -d "data" ] && [ -z "$(ls -A data 2>/dev/null)" ]; then
    rmdir data 2>/dev/null || true
fi

echo "✅ Cleanup completed!"
echo ""
echo "You can now run:"
echo "  mvn test          - Run tests with in-memory database"
echo "  mvn spring-boot:run - Start application with file database"