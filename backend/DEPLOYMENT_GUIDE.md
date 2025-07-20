# Smart Shoe Complete Deployment Guide

Complete automation scripts for deploying the Smart Shoe application to EC2.

## 🚀 Quick Start

### 1. Deploy Backend + Nginx Setup
```bash
# Deploy everything with one command
./deploy-to-ec2.sh [EC2_IP] [PEM_FILE]

# Example:
./deploy-to-ec2.sh 13.201.120.175 ~/.ssh/my-key.pem
```

### 2. Deploy Frontend (Optional)
```bash
# Deploy React frontend
./deploy-frontend-to-ec2.sh [EC2_IP] [PEM_FILE]
```

## 📋 What the Scripts Do

### Backend Deployment (`deploy-to-ec2.sh`)

#### ✅ Complete Automation:
1. **Pre-deployment**:
   - ✅ Checks Maven, PEM file, permissions
   - ✅ Cleans up local H2 database files
   - ✅ Builds Spring Boot JAR file

2. **EC2 Setup**:
   - ✅ Creates service directories (`/opt/smartshoe-backend/`)
   - ✅ Uploads JAR file and configuration
   - ✅ Creates systemd service for auto-start

3. **Nginx Configuration**:
   - ✅ Installs Nginx (if needed)
   - ✅ Configures reverse proxy for API endpoints
   - ✅ Sets up CORS headers for frontend communication
   - ✅ Handles preflight OPTIONS requests

4. **Security & Firewall**:
   - ✅ Opens required ports (80, 443, 8080)
   - ✅ Sets up security headers
   - ✅ Configures proper file permissions

5. **Service Management**:
   - ✅ Starts Spring Boot as systemd service
   - ✅ Enables auto-restart on failure
   - ✅ Creates management scripts

6. **Verification**:
   - ✅ Tests health endpoints
   - ✅ Shows service logs
   - ✅ Provides troubleshooting info

### Frontend Deployment (`deploy-frontend-to-ec2.sh`)

#### ✅ Frontend Automation:
1. **Build Process**:
   - ✅ Installs npm dependencies
   - ✅ Builds React app for production
   - ✅ Optimizes assets

2. **Deployment**:
   - ✅ Creates web directory (`/var/www/html/`)
   - ✅ Uploads built files via rsync
   - ✅ Sets proper permissions for Nginx

## 🛠️ Prerequisites

### On Your Local Machine:
- ✅ Maven installed
- ✅ Node.js and npm (for frontend)
- ✅ SSH access to EC2 instance
- ✅ PEM key file with proper permissions (400)

### On EC2 Instance:
- ✅ Amazon Linux 2 or CentOS/RHEL
- ✅ Java 17+ (will be installed if missing)
- ✅ sudo access for ec2-user

## 📁 File Structure After Deployment

```
EC2 Instance:
├── /opt/smartshoe-backend/
│   ├── api-3.0.0.jar              # Spring Boot application
│   ├── application.yml            # Configuration
│   ├── logs/                      # Application logs
│   ├── data/                      # H2 database files
│   └── manage.sh                  # Management script
├── /var/www/html/                 # Frontend files (if deployed)
├── /etc/systemd/system/
│   └── smartshoe-backend.service  # Systemd service
└── /etc/nginx/conf.d/
    └── smartshoe-backend.conf     # Nginx configuration
```

## 🔧 Management Commands

### On EC2 Instance:
```bash
# Service management
/opt/smartshoe-backend/manage.sh start      # Start service
/opt/smartshoe-backend/manage.sh stop       # Stop service
/opt/smartshoe-backend/manage.sh restart    # Restart service
/opt/smartshoe-backend/manage.sh status     # Check status
/opt/smartshoe-backend/manage.sh logs       # Follow logs
/opt/smartshoe-backend/manage.sh cleanup    # Clean H2 files

# Direct systemd commands
sudo systemctl start smartshoe-backend
sudo systemctl stop smartshoe-backend
sudo systemctl restart smartshoe-backend
sudo systemctl status smartshoe-backend

# View logs
sudo journalctl -u smartshoe-backend -f     # Follow logs
sudo journalctl -u smartshoe-backend -n 50  # Recent logs
```

## 🌐 Endpoints After Deployment

### Backend API:
- **Base URL**: `http://YOUR_EC2_IP/api/`
- **Health Check**: `http://YOUR_EC2_IP/health`
- **H2 Console**: `http://YOUR_EC2_IP/api/h2-console` (dev only)

### Frontend (if deployed):
- **Web App**: `http://YOUR_EC2_IP/`

### Example Endpoints:
```bash
# Test API health
curl http://13.201.120.175/health

# Test API endpoints
curl http://13.201.120.175/api/health/status
curl http://13.201.120.175/api/credentials/test
```

## 🚨 Troubleshooting

### Common Issues:

#### 1. **PEM File Permission Error**
```bash
chmod 400 ~/.ssh/your-key.pem
```

#### 2. **Service Won't Start**
```bash
# Check logs
sudo journalctl -u smartshoe-backend -n 50

# Check Java version
java -version

# Check port conflicts
sudo netstat -tlnp | grep 8080
```

#### 3. **Database Lock Issues**
```bash
# Run cleanup script
/opt/smartshoe-backend/manage.sh cleanup

# Restart service
/opt/smartshoe-backend/manage.sh restart
```

#### 4. **CORS Issues**
- ✅ Already handled by Nginx configuration
- ✅ Supports all HTTP methods
- ✅ Handles preflight requests

#### 5. **Nginx Issues**
```bash
# Check Nginx status
sudo systemctl status nginx

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

## 🔄 Redeployment

### Update Backend:
```bash
# Just run the deployment script again
./deploy-to-ec2.sh
```

### Update Frontend:
```bash
# Deploy new frontend build
./deploy-frontend-to-ec2.sh
```

### Update Configuration:
```bash
# Edit application.yml locally, then redeploy
./deploy-to-ec2.sh
```

## 🔒 Security Features

### ✅ Built-in Security:
- **Firewall**: Configured for HTTP/HTTPS
- **Service Isolation**: Runs as non-root user
- **File Permissions**: Properly secured
- **CORS**: Configured for frontend access
- **Security Headers**: X-Frame-Options, XSS protection
- **Process Management**: Systemd with restart policies

## 📊 Monitoring

### Service Health:
```bash
# Check if service is running
systemctl is-active smartshoe-backend

# Check service uptime
systemctl status smartshoe-backend

# Monitor resource usage
top -p $(pgrep -f smartshoe)
```

### Application Health:
```bash
# Health endpoint
curl http://YOUR_EC2_IP/health

# Application metrics (if enabled)
curl http://YOUR_EC2_IP/api/actuator/metrics
```

## 📞 Support

### Log Locations:
- **Application Logs**: `sudo journalctl -u smartshoe-backend`
- **Nginx Logs**: `/var/log/nginx/access.log`, `/var/log/nginx/error.log`
- **System Logs**: `/var/log/messages`

### Key Configuration Files:
- **Service**: `/etc/systemd/system/smartshoe-backend.service`
- **Nginx**: `/etc/nginx/conf.d/smartshoe-backend.conf`
- **Application**: `/opt/smartshoe-backend/application.yml`

---

## 🎉 Success!

After running the deployment script, your Smart Shoe application will be:
- ✅ **Fully deployed** on EC2
- ✅ **Auto-starting** on boot
- ✅ **Load balanced** with Nginx
- ✅ **CORS enabled** for frontend
- ✅ **Health monitored** with endpoints
- ✅ **Production ready** with proper logging

Your application is now live at `http://YOUR_EC2_IP`! 🚀