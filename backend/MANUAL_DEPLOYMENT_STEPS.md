# Manual Step-by-Step Deployment Guide for EC2

## 🎯 Goal: Deploy Smart Shoe Backend to EC2 IP: 13.201.120.175

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ EC2 instance running (13.201.120.175)
- ✅ SSH access with PEM key
- ✅ Maven installed locally
- ✅ Java 17+ on EC2 instance

---

## 🔨 STEP 1: Build the Application Locally

### 1.1 Clean up local database files
```bash
cd /mnt/d/project/diabetic-smart-shoe/backend

# Kill any running Java processes
pkill -f java || true
pkill -f smartshoe || true

# Remove H2 database files
find . -name "*.mv.db" -delete 2>/dev/null || true
find . -name "*.trace.db" -delete 2>/dev/null || true
find . -name "*.lock.db" -delete 2>/dev/null || true
```

### 1.2 Build the JAR file
```bash
# Clean and build (skip tests to avoid database issues)
mvn clean package -DskipTests=true

# Verify JAR was created
ls -la target/api-3.0.0.jar
```

**✅ Expected Result**: You should see `target/api-3.0.0.jar` file created

---

## 🚀 STEP 2: Connect to EC2 and Prepare Directories

### 2.1 SSH to your EC2 instance
```bash
# Replace with your actual PEM file path
ssh -i ~/.ssh/your-key.pem ec2-user@13.201.120.175
```

### 2.2 Create application directories
```bash
# Create application directory
sudo mkdir -p /opt/smartshoe-backend
sudo chown ec2-user:ec2-user /opt/smartshoe-backend

# Create subdirectories
mkdir -p /opt/smartshoe-backend/logs
mkdir -p /opt/smartshoe-backend/data
```

### 2.3 Install Java (if not installed)
```bash
# Check Java version
java -version

# If Java not installed or wrong version:
sudo yum update -y
sudo yum install -y java-17-openjdk
```

**✅ Expected Result**: Java 17+ should be installed and working

---

## 📤 STEP 3: Upload Files to EC2

### 3.1 Open a new terminal (keep SSH session open) and upload JAR
```bash
# From your local machine (new terminal)
cd /mnt/d/project/diabetic-smart-shoe/backend

# Upload JAR file
scp -i ~/.ssh/your-key.pem target/api-3.0.0.jar ec2-user@13.201.120.175:/opt/smartshoe-backend/

# Upload configuration
scp -i ~/.ssh/your-key.pem src/main/resources/application.yml ec2-user@13.201.120.175:/opt/smartshoe-backend/
```

### 3.2 Verify files uploaded (in SSH session)
```bash
ls -la /opt/smartshoe-backend/
```

**✅ Expected Result**: You should see `api-3.0.0.jar` and `application.yml` files

---

## ⚙️ STEP 4: Create Systemd Service

### 4.1 Create service file (in SSH session)
```bash
sudo tee /etc/systemd/system/smartshoe-backend.service > /dev/null << 'EOF'
[Unit]
Description=Smart Shoe Backend API
After=network.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=/opt/smartshoe-backend
ExecStart=/usr/bin/java -jar -Dspring.profiles.active=production /opt/smartshoe-backend/api-3.0.0.jar
ExecStop=/bin/kill -15 $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=smartshoe-backend

# Environment variables
Environment=JAVA_OPTS=-Xmx512m -Xms256m
Environment=SERVER_PORT=8080

[Install]
WantedBy=multi-user.target
EOF
```

### 4.2 Enable and start the service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable smartshoe-backend

# Start the service
sudo systemctl start smartshoe-backend

# Check status
sudo systemctl status smartshoe-backend
```

**✅ Expected Result**: Service should be "active (running)"

### 4.3 Check application logs
```bash
# View recent logs
sudo journalctl -u smartshoe-backend -n 50

# Follow logs (Ctrl+C to exit)
sudo journalctl -u smartshoe-backend -f
```

**✅ Expected Result**: Application should start without errors

---

## 🌐 STEP 5: Install and Configure Nginx

### 5.1 Install Nginx
```bash
# Install Nginx
sudo yum install -y nginx

# Start and enable Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

### 5.2 Create Nginx configuration for your app
```bash
sudo tee /etc/nginx/conf.d/smartshoe-backend.conf > /dev/null << 'EOF'
server {
    listen 80;
    server_name 13.201.120.175;
    
    # API endpoints with CORS
    location /api/ {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CORS headers for specific origin
        add_header 'Access-Control-Allow-Origin' 'http://13.201.120.175' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,Accept,Origin,Access-Control-Request-Method,Access-Control-Request-Headers' always;
        add_header 'Access-Control-Allow-Credentials' 'true' always;
        add_header 'Access-Control-Expose-Headers' 'Authorization,Content-Disposition' always;
        
        # Handle preflight OPTIONS requests
        if ($request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' 'http://13.201.120.175' always;
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
        proxy_set_header Host $host;
        
        # CORS for health endpoint
        add_header 'Access-Control-Allow-Origin' 'http://13.201.120.175' always;
        add_header 'Access-Control-Allow-Methods' 'GET, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
    }
    
    # Frontend static files
    location / {
        root /var/www/html;
        try_files $uri $uri/ /index.html;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
    }
}
EOF
```

### 5.3 Test and restart Nginx
```bash
# Test configuration
sudo nginx -t

# If test passes, restart Nginx
sudo systemctl restart nginx

# Check Nginx status
sudo systemctl status nginx
```

**✅ Expected Result**: Nginx should be running with no configuration errors

---

## 🔥 STEP 6: Configure Firewall

### 6.1 Open required ports
```bash
# For systems with firewalld
sudo firewall-cmd --permanent --add-service=http 2>/dev/null || true
sudo firewall-cmd --permanent --add-service=https 2>/dev/null || true
sudo firewall-cmd --permanent --add-port=8080/tcp 2>/dev/null || true
sudo firewall-cmd --reload 2>/dev/null || true

# For systems with ufw (alternative)
sudo ufw allow 80/tcp 2>/dev/null || true
sudo ufw allow 443/tcp 2>/dev/null || true
sudo ufw allow 8080/tcp 2>/dev/null || true
```

**Note**: This might show errors if firewall is not configured - that's usually OK on EC2.

---

## 🧪 STEP 7: Test Your Deployment

### 7.1 Test from EC2 instance (in SSH session)
```bash
# Test Spring Boot directly
curl http://localhost:8080/actuator/health

# Test through Nginx
curl http://localhost/health

# Test API endpoint
curl http://localhost/api/health/status
```

### 7.2 Test from your local machine (new terminal)
```bash
# Test health endpoint
curl http://13.201.120.175/health

# Test API endpoint
curl http://13.201.120.175/api/health/status

# Test CORS preflight
curl -H "Origin: http://13.201.120.175" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type,Authorization" \
     -X OPTIONS \
     http://13.201.120.175/api/health/status
```

**✅ Expected Result**: All commands should return successful responses

---

## 🌐 STEP 8: Create CORS Test Page

### 8.1 Create test page (in SSH session)
```bash
# Create web directory
sudo mkdir -p /var/www/html
sudo chown -R ec2-user:ec2-user /var/www/html

# Create CORS test page
cat > /var/www/html/cors-test.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>CORS Test - Smart Shoe</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .result { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Smart Shoe CORS Test</h1>
    <p>Testing CORS connectivity to backend API at 13.201.120.175</p>
    
    <button onclick="testHealth()">Test Health</button>
    <button onclick="testAPI()">Test API</button>
    
    <div id="results"></div>
    
    <script>
        function addResult(message, success = true) {
            const div = document.createElement('div');
            div.className = 'result ' + (success ? 'success' : 'error');
            div.innerHTML = new Date().toLocaleTimeString() + ': ' + message;
            document.getElementById('results').appendChild(div);
        }
        
        async function testHealth() {
            try {
                const response = await fetch('http://13.201.120.175/health');
                const data = await response.text();
                addResult('Health: ' + response.status + ' - ' + data.substring(0, 100), response.ok);
            } catch (error) {
                addResult('Health Error: ' + error.message, false);
            }
        }
        
        async function testAPI() {
            try {
                const response = await fetch('http://13.201.120.175/api/health/status');
                const data = await response.text();
                addResult('API: ' + response.status + ' - ' + data, response.ok);
            } catch (error) {
                addResult('API Error: ' + error.message, false);
            }
        }
        
        // Auto-test on load
        window.onload = function() {
            setTimeout(testHealth, 1000);
            setTimeout(testAPI, 2000);
        };
    </script>
</body>
</html>
EOF

# Set proper permissions
sudo chown -R nginx:nginx /var/www/html 2>/dev/null || sudo chown -R apache:apache /var/www/html 2>/dev/null || true
sudo chmod -R 755 /var/www/html
```

### 8.2 Test the CORS test page
```bash
# From your local machine browser, open:
# http://13.201.120.175/cors-test.html
```

**✅ Expected Result**: Page should load and show successful green test results

---

## 🔧 STEP 9: Create Management Script

### 9.1 Create management script (in SSH session)
```bash
cat > /opt/smartshoe-backend/manage.sh << 'EOF'
#!/bin/bash

SERVICE_NAME="smartshoe-backend"

case "$1" in
    start)
        echo "Starting $SERVICE_NAME..."
        sudo systemctl start $SERVICE_NAME
        ;;
    stop)
        echo "Stopping $SERVICE_NAME..."
        sudo systemctl stop $SERVICE_NAME
        ;;
    restart)
        echo "Restarting $SERVICE_NAME..."
        sudo systemctl restart $SERVICE_NAME
        ;;
    status)
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    logs)
        sudo journalctl -u $SERVICE_NAME -f
        ;;
    logs-recent)
        sudo journalctl -u $SERVICE_NAME --no-pager -n 50
        ;;
    cleanup)
        echo "Cleaning up H2 database files..."
        find /opt/$SERVICE_NAME -name "*.mv.db" -delete 2>/dev/null || true
        find /opt/$SERVICE_NAME -name "*.trace.db" -delete 2>/dev/null || true
        find /opt/$SERVICE_NAME -name "*.lock.db" -delete 2>/dev/null || true
        echo "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|logs-recent|cleanup}"
        exit 1
        ;;
esac
EOF

# Make it executable
chmod +x /opt/smartshoe-backend/manage.sh
```

---

## ✅ STEP 10: Verify Everything is Working

### 10.1 Check all services
```bash
# Check backend service
sudo systemctl status smartshoe-backend

# Check Nginx
sudo systemctl status nginx

# Check if ports are listening
sudo netstat -tlnp | grep :80
sudo netstat -tlnp | grep :8080
```

### 10.2 Test all endpoints
```bash
# From local machine:
curl http://13.201.120.175/health
curl http://13.201.120.175/api/health/status

# Open in browser:
# http://13.201.120.175/cors-test.html
```

**✅ Expected Results**:
- Backend service: `active (running)`
- Nginx service: `active (running)`
- Port 80: Nginx listening
- Port 8080: Spring Boot listening
- All curl commands return successful responses
- CORS test page shows green success messages

---

## 🎉 SUCCESS! Your Application is Now Running

### Your endpoints:
- **Frontend/Test Page**: `http://13.201.120.175/cors-test.html`
- **API Health**: `http://13.201.120.175/api/health/status`
- **System Health**: `http://13.201.120.175/health`

### Management commands (on EC2):
```bash
/opt/smartshoe-backend/manage.sh start      # Start service
/opt/smartshoe-backend/manage.sh stop       # Stop service
/opt/smartshoe-backend/manage.sh restart    # Restart service
/opt/smartshoe-backend/manage.sh status     # Check status
/opt/smartshoe-backend/manage.sh logs       # View logs
/opt/smartshoe-backend/manage.sh cleanup    # Clean database files
```

---

## 🚨 Troubleshooting

### If service won't start:
```bash
# Check logs
sudo journalctl -u smartshoe-backend -n 50

# Check Java version
java -version

# Clean database files and restart
/opt/smartshoe-backend/manage.sh cleanup
/opt/smartshoe-backend/manage.sh restart
```

### If CORS is not working:
```bash
# Check Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Check Nginx logs
sudo tail -f /var/log/nginx/error.log
```

### If can't access from outside:
- Check AWS Security Groups (allow HTTP port 80)
- Check if EC2 instance has public IP
- Verify firewall settings

---

This manual process should get your Smart Shoe backend deployed and working with proper CORS configuration! 🚀