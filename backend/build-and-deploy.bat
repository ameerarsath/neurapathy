@echo off
REM ====================================================================
REM WINDOWS PRODUCTION BUILD AND DEPLOY SCRIPT FOR SMART SHOE API
REM Builds Docker image and deploys to production environment
REM ====================================================================

setlocal enabledelayedexpansion

REM ====================================================================
REM CONFIGURATION
REM ====================================================================
set IMAGE_NAME=smartshoe/api
set IMAGE_TAG=3.0.0
set LATEST_TAG=latest
set PROJECT_NAME=smartshoe-prod
set COMPOSE_FILE=docker-compose.prod.yml

REM Colors (Windows compatible)
set COLOR_INFO=96
set COLOR_SUCCESS=92
set COLOR_WARNING=93
set COLOR_ERROR=91
set COLOR_RESET=0

REM ====================================================================
REM FUNCTIONS
REM ====================================================================
:log_info
    echo [INFO] %~1
    goto :eof

:log_success
    echo [SUCCESS] %~1
    goto :eof

:log_warning
    echo [WARNING] %~1
    goto :eof

:log_error
    echo [ERROR] %~1
    goto :eof

:show_help
    echo.
    echo Smart Shoe API Production Build and Deploy Script
    echo.
    echo USAGE:
    echo     %~nx0 [COMMAND]
    echo.
    echo COMMANDS:
    echo     build       Build Docker image only
    echo     deploy      Deploy with docker-compose
    echo     full        Build and deploy (default)
    echo     clean       Clean Docker resources
    echo     status      Show deployment status
    echo     logs        Show service logs
    echo     stop        Stop all services
    echo     help        Show this help
    echo.
    goto :eof

:check_prerequisites
    call :log_info "Checking prerequisites..."
    
    REM Check Docker
    docker --version >nul 2>&1
    if !errorlevel! neq 0 (
        call :log_error "Docker is not installed or not in PATH"
        exit /b 1
    )
    
    REM Check Docker Compose
    docker-compose --version >nul 2>&1
    if !errorlevel! neq 0 (
        call :log_error "Docker Compose is not installed or not in PATH"
        exit /b 1
    )
    
    REM Check if Docker is running
    docker info >nul 2>&1
    if !errorlevel! neq 0 (
        call :log_error "Docker is not running. Please start Docker Desktop."
        exit /b 1
    )
    
    REM Check files
    if not exist "Dockerfile" (
        call :log_error "Dockerfile not found"
        exit /b 1
    )
    
    if not exist "pom.xml" (
        call :log_error "pom.xml not found"
        exit /b 1
    )
    
    if not exist "%COMPOSE_FILE%" (
        call :log_error "Docker Compose file not found: %COMPOSE_FILE%"
        exit /b 1
    )
    
    call :log_success "Prerequisites check passed"
    goto :eof

:build_docker_image
    call :log_info "Building Docker image..."
    
    set BUILD_TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%-%time:~0,2%%time:~3,2%%time:~6,2%
    set BUILD_TIMESTAMP=%BUILD_TIMESTAMP: =0%
    
    docker build ^
        --target production ^
        --tag "%IMAGE_NAME%:%IMAGE_TAG%" ^
        --tag "%IMAGE_NAME%:%LATEST_TAG%" ^
        --tag "%IMAGE_NAME%:%BUILD_TIMESTAMP%" ^
        --label "org.opencontainers.image.created=%date% %time%" ^
        --label "org.opencontainers.image.version=%IMAGE_TAG%" ^
        --label "org.opencontainers.image.title=Smart Shoe API" ^
        .
    
    if !errorlevel! neq 0 (
        call :log_error "Docker build failed"
        exit /b 1
    )
    
    call :log_success "Docker image built successfully"
    
    REM Show image info
    for /f "tokens=*" %%i in ('docker image inspect "%IMAGE_NAME%:%IMAGE_TAG%" --format="{{.Size}}"') do set IMAGE_SIZE=%%i
    call :log_info "Image size: %IMAGE_SIZE% bytes"
    
    goto :eof

:create_environment_file
    if not exist ".env" (
        call :log_info "Creating environment file..."
        
        echo # Smart Shoe API Production Environment > .env
        echo # Generated on %date% %time% >> .env
        echo. >> .env
        echo # Database Configuration >> .env
        echo DB_PASSWORD=SmartShoe2024! >> .env
        echo POSTGRES_DB=smartshoe_prod >> .env
        echo POSTGRES_USER=smartshoe_user >> .env
        echo. >> .env
        echo # Redis Configuration >> .env
        echo REDIS_PASSWORD=RedisSmartShoe2024! >> .env
        echo. >> .env
        echo # JWT Configuration >> .env
        echo JWT_SECRET=YourSuperSecretJWTKeyForProductionUse2024! >> .env
        echo JWT_EXPIRATION=86400000 >> .env
        echo. >> .env
        echo # Grafana Configuration >> .env
        echo GRAFANA_PASSWORD=GrafanaSmartShoe2024! >> .env
        echo. >> .env
        echo # Optional Email Configuration >> .env
        echo EMAIL_ENABLED=false >> .env
        echo EMAIL_FROM=noreply@smartshoe.com >> .env
        echo SMTP_HOST=smtp.gmail.com >> .env
        echo SMTP_PORT=587 >> .env
        echo SMTP_USERNAME= >> .env
        echo SMTP_PASSWORD= >> .env
        
        call :log_success "Environment file created: .env"
        call :log_warning "Please review and update the .env file with your production values"
    ) else (
        call :log_info "Environment file already exists: .env"
    )
    goto :eof

:create_directories
    call :log_info "Creating necessary directories..."
    
    if not exist "docker\nginx\conf.d" mkdir "docker\nginx\conf.d"
    if not exist "docker\nginx\ssl" mkdir "docker\nginx\ssl"
    if not exist "docker\postgres\init" mkdir "docker\postgres\init"
    if not exist "docker\prometheus" mkdir "docker\prometheus"
    if not exist "docker\grafana\dashboards" mkdir "docker\grafana\dashboards"
    if not exist "docker\grafana\datasources" mkdir "docker\grafana\datasources"
    if not exist "backups" mkdir "backups"
    if not exist "logs" mkdir "logs"
    
    call :log_success "Directories created"
    goto :eof

:deploy_stack
    call :log_info "Deploying Smart Shoe production stack..."
    
    REM Pull latest images
    call :log_info "Pulling latest images..."
    docker-compose -f "%COMPOSE_FILE%" -p "%PROJECT_NAME%" pull
    
    REM Deploy the stack
    call :log_info "Starting services..."
    docker-compose -f "%COMPOSE_FILE%" -p "%PROJECT_NAME%" up -d
    
    if !errorlevel! neq 0 (
        call :log_error "Deployment failed"
        exit /b 1
    )
    
    REM Wait for services to start
    call :log_info "Waiting for services to start..."
    timeout /t 30 /nobreak >nul
    
    call :log_success "Smart Shoe stack deployed successfully!"
    call :show_service_urls
    goto :eof

:show_service_urls
    echo.
    echo ========================================
    echo SERVICE URLS
    echo ========================================
    echo Smart Shoe API:     http://localhost:8080
    echo API Health:         http://localhost:8080/actuator/health
    echo API Metrics:        http://localhost:8080/actuator/metrics
    echo PostgreSQL:         localhost:5432
    echo Redis:              localhost:6379
    echo Prometheus:         http://localhost:9090
    echo Grafana:            http://localhost:3000
    echo ========================================
    echo.
    goto :eof

:show_status
    call :log_info "Smart Shoe stack status:"
    docker-compose -f "%COMPOSE_FILE%" -p "%PROJECT_NAME%" ps
    echo.
    
    call :log_info "Container health status:"
    docker ps --filter "name=smartshoe" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    goto :eof

:show_logs
    call :log_info "Showing service logs (last 100 lines)..."
    docker-compose -f "%COMPOSE_FILE%" -p "%PROJECT_NAME%" logs --tail=100
    goto :eof

:stop_services
    call :log_info "Stopping Smart Shoe services..."
    docker-compose -f "%COMPOSE_FILE%" -p "%PROJECT_NAME%" stop
    call :log_success "Services stopped"
    goto :eof

:clean_docker
    call :log_info "Cleaning Docker resources..."
    
    echo This will remove stopped containers and unused images.
    set /p "confirm=Do you want to continue? (y/N): "
    if /i not "!confirm!"=="y" (
        call :log_info "Operation cancelled"
        goto :eof
    )
    
    REM Remove stopped containers
    docker container prune -f
    
    REM Remove unused images
    docker image prune -f
    
    call :log_success "Cleanup completed"
    goto :eof

REM ====================================================================
REM MAIN EXECUTION
REM ====================================================================
if "%~1"=="" set COMMAND=full
if "%~1"=="build" set COMMAND=build
if "%~1"=="deploy" set COMMAND=deploy
if "%~1"=="full" set COMMAND=full
if "%~1"=="clean" set COMMAND=clean
if "%~1"=="status" set COMMAND=status
if "%~1"=="logs" set COMMAND=logs
if "%~1"=="stop" set COMMAND=stop
if "%~1"=="help" set COMMAND=help

echo ====================================================================
echo    SMART SHOE API - PRODUCTION BUILD AND DEPLOY
echo ====================================================================
echo.

if "%COMMAND%"=="help" (
    call :show_help
    goto :end
)

if "%COMMAND%"=="status" (
    call :show_status
    goto :end
)

if "%COMMAND%"=="logs" (
    call :show_logs
    goto :end
)

if "%COMMAND%"=="stop" (
    call :stop_services
    goto :end
)

if "%COMMAND%"=="clean" (
    call :clean_docker
    goto :end
)

REM For build, deploy, and full commands, check prerequisites
call :check_prerequisites
if !errorlevel! neq 0 goto :error

if "%COMMAND%"=="build" (
    call :build_docker_image
    if !errorlevel! neq 0 goto :error
    goto :success
)

if "%COMMAND%"=="deploy" (
    call :create_directories
    call :create_environment_file
    call :deploy_stack
    if !errorlevel! neq 0 goto :error
    goto :success
)

if "%COMMAND%"=="full" (
    call :build_docker_image
    if !errorlevel! neq 0 goto :error
    
    call :create_directories
    call :create_environment_file
    call :deploy_stack
    if !errorlevel! neq 0 goto :error
    goto :success
)

call :log_error "Unknown command: %COMMAND%"
call :show_help
goto :error

:success
echo.
echo ====================================================================
echo     BUILD AND DEPLOYMENT COMPLETED SUCCESSFULLY!
echo ====================================================================
echo.
call :log_success "Smart Shoe API is ready for production use!"
echo.
call :log_info "Next steps:"
echo   1. Test the API: http://localhost:8080/actuator/health
echo   2. Access Grafana dashboard: http://localhost:3000
echo   3. Monitor with Prometheus: http://localhost:9090
echo.
goto :end

:error
echo.
echo ====================================================================
echo     BUILD AND DEPLOYMENT FAILED!
echo ====================================================================
echo.
call :log_error "Please check the error messages above and try again."
exit /b 1

:end
pause