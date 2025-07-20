# WebSocket Deployment Instructions

## Backend Changes Made
The backend now includes WebSocket support with the following additions:

### 1. Dependencies Added
- `spring-boot-starter-websocket` in `pom.xml`

### 2. New Configuration Files
- `WebSocketConfig.java` - WebSocket endpoint configuration
- `SmartShoeWebSocketHandler.java` - Message handling and broadcasting
- `WebSocketTestController.java` - REST API for testing WebSocket functionality

### 3. Security Updates
- Updated `WebSecurityConfig.java` to allow WebSocket endpoints (`/ws/**`, `/websocket/**`)

## Deployment Steps

### Step 1: Build Updated Backend
```bash
cd /mnt/d/project/diabetic-smart-shoe/backend
mvn clean package -DskipTests
```

### Step 2: Upload to EC2 Server
```bash
# Replace with your actual key path
scp -i ~/.ssh/your-key.pem target/api-3.0.0.jar ec2-user@13.201.120.175:~/
```

### Step 3: Stop Current Service
```bash
ssh -i ~/.ssh/your-key.pem ec2-user@13.201.120.175
sudo systemctl stop smartshoe-backend
```

### Step 4: Replace JAR File
```bash
sudo mv ~/api-3.0.0.jar /opt/smartshoe/smartshoe-backend.jar
sudo chown smartshoe:smartshoe /opt/smartshoe/smartshoe-backend.jar
```

### Step 5: Start Service
```bash
sudo systemctl start smartshoe-backend
sudo systemctl status smartshoe-backend
```

### Step 6: Verify WebSocket is Working
```bash
# Check WebSocket status endpoint
curl http://13.201.120.175:8080/api/websocket/status

# Test WebSocket broadcast
curl -X POST http://13.201.120.175:8080/api/websocket/broadcast/test \
  -H "Content-Type: application/json" \
  -d '{"message": "WebSocket test from deployment"}'
```

## Frontend Changes Made
- Replaced Socket.IO with native WebSocket API
- Updated connection handling and message processing
- Added development mode WebSocket disable (to prevent connection errors during development)

## Testing WebSocket Connection

### 1. Using Browser Console
```javascript
// Enable WebSocket for testing
localStorage.removeItem('disableWebSocket');
window.location.reload();
```

### 2. Using Test Files
- Open `frontend/websocket-test.html` in browser
- Open `frontend/test-ws-connection.html` for simple test

### 3. WebSocket Endpoints Available
- **WebSocket Connection**: `ws://13.201.120.175:8080/ws`
- **Status API**: `GET /api/websocket/status`
- **Test Broadcast**: `POST /api/websocket/broadcast/test`
- **Device Alert**: `POST /api/websocket/broadcast/device-alert`
- **Medical Alert**: `POST /api/websocket/broadcast/medical-alert`

## Troubleshooting

### If WebSocket still fails after deployment:
1. Check if backend service is running: `sudo systemctl status smartshoe-backend`
2. Check logs: `sudo journalctl -u smartshoe-backend -f`
3. Verify port 8080 is open: `sudo netstat -tlnp | grep 8080`
4. Test REST endpoints first: `curl http://13.201.120.175:8080/api/health`

### Common Issues:
- **Code 1006**: Connection refused - backend not running or WebSocket not configured
- **CORS errors**: Check if CORS configuration includes WebSocket origins
- **Connection timeouts**: Check firewall settings and security groups

## Current Status
- ✅ Backend WebSocket code implemented and compiled
- ⚠️ Backend deployment needed on production server
- ✅ Frontend updated to use native WebSocket
- ✅ Development mode WebSocket disabled to prevent errors