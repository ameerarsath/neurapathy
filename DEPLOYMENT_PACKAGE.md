# 🚀 Smart Shoe Deployment Package

## ✅ Build Files Ready

### Frontend Build Files
- **Location**: `/mnt/d/project/diabetic-smart-shoe/frontend/dist/`
- **Main File**: `dist/index.html` (Production-ready frontend)
- **API Test**: `dist/api-test.html` (API connectivity tester)
- **Status**: ✅ Built successfully with WebSocket disabled

### Backend Build Files  
- **Location**: `/mnt/d/project/diabetic-smart-shoe/backend/target/`
- **JAR File**: `api-3.0.0.jar` (85MB - includes WebSocket support)
- **Status**: ✅ Built successfully with WebSocket configuration

## 🔧 Deployment Instructions

### 1. Deploy Backend (Updated JAR with WebSocket)

```bash
# Upload new JAR to server
scp -i ~/.ssh/your-key.pem backend/target/api-3.0.0.jar ec2-user@13.201.120.175:~/

# SSH into server
ssh -i ~/.ssh/your-key.pem ec2-user@13.201.120.175

# Stop current service
sudo systemctl stop smartshoe-backend

# Replace JAR file
sudo mv ~/api-3.0.0.jar /opt/smartshoe/smartshoe-backend.jar
sudo chown smartshoe:smartshoe /opt/smartshoe/smartshoe-backend.jar

# Start service
sudo systemctl start smartshoe-backend
sudo systemctl status smartshoe-backend

# Verify deployment
curl http://13.201.120.175:8080/api/health
curl http://13.201.120.175:8080/api/websocket/status
```

### 2. Deploy Frontend

```bash
# Upload frontend files
scp -i ~/.ssh/your-key.pem -r frontend/dist/* ec2-user@13.201.120.175:/var/www/html/

# Or using rsync
rsync -avz --delete -e "ssh -i ~/.ssh/your-key.pem" frontend/dist/ ec2-user@13.201.120.175:/var/www/html/

# Set proper permissions
ssh -i ~/.ssh/your-key.pem ec2-user@13.201.120.175
sudo chown -R nginx:nginx /var/www/html
sudo chmod -R 755 /var/www/html
```

## 🧪 Testing After Deployment

### 1. Test Backend API
```bash
curl http://13.201.120.175:8080/api/health
# Expected: {"message":"Smart Shoe Backend is running successfully!","status":"UP"}

curl http://13.201.120.175:8080/api/websocket/status  
# Expected: {"status":"active","activeConnections":0}
```

### 2. Test Frontend
- **Main App**: http://13.201.120.175/
- **API Tester**: http://13.201.120.175/api-test.html

### 3. Enable WebSocket (After Backend Deployment)
```javascript
// In browser console on frontend
localStorage.removeItem('disableWebSocket');
window.location.reload();
```

## 🔄 WebSocket Status

### Current Status: ⚠️ Temporarily Disabled
- **Reason**: Prevents connection errors while backend is being deployed
- **Frontend**: WebSocket disabled in code (`disableWebSocket = true`)
- **Backend**: WebSocket implementation ready, needs deployment

### After Backend Deployment: ✅ Enable WebSocket
1. **Update frontend code**:
   ```javascript
   // In WebSocketContext.jsx, change line 62:
   const disableWebSocket = false // Enable WebSocket
   ```
2. **Rebuild frontend** with WebSocket enabled
3. **Deploy updated frontend**

## 📦 Package Contents

### Backend Features
- ✅ Spring Boot 3.2.0 with Java 17
- ✅ H2 Database with sample data
- ✅ REST API with comprehensive endpoints
- ✅ Security configuration with CORS
- ✅ WebSocket support (SmartShoeWebSocketHandler)
- ✅ ML API integration ready
- ✅ Authentication system
- ✅ Health monitoring endpoints

### Frontend Features  
- ✅ Production-ready React application
- ✅ API connectivity testing
- ✅ Responsive design
- ✅ Error handling
- ✅ WebSocket context (disabled for now)
- ✅ Environment-specific configuration

## 🚨 Important Notes

1. **WebSocket Errors Fixed**: No more console spam - WebSocket is cleanly disabled
2. **Backend Updated**: New JAR includes WebSocket configuration
3. **Production Ready**: Both frontend and backend configured for production server
4. **CORS Configured**: All origins allowed for development/testing
5. **Database Ready**: H2 with sample patients, devices, and medical readings

## 📋 Verification Checklist

After deployment, verify:
- [ ] Backend health endpoint responds
- [ ] Frontend loads correctly
- [ ] API test page works
- [ ] No WebSocket errors in console
- [ ] Database endpoints accessible
- [ ] CORS working properly

## 🔧 Troubleshooting

### If backend fails to start:
```bash
sudo journalctl -u smartshoe-backend -f
```

### If frontend not accessible:
```bash
sudo systemctl status nginx
sudo nginx -t
```

### If API calls fail:
- Check CORS configuration
- Verify backend is running on port 8080
- Test with curl first

---

**Status**: ✅ Ready for deployment
**WebSocket**: ⚠️ Disabled (will enable after backend deployment)
**Build Date**: $(date)