# CORS Configuration Summary for EC2 IP: 13.201.120.175

## 🎯 Configuration Overview

Your Smart Shoe application is now configured with comprehensive CORS support for both frontend and backend communication using your public EC2 IP: **13.201.120.175**.

## ✅ What's Configured

### 1. **Spring Boot Backend CORS** (`CorsConfig.java`)
```java
// Allowed origins include:
"http://13.201.120.175"        // Your public IP (HTTP)
"https://13.201.120.175"       // Your public IP (HTTPS)
"http://13.201.120.175:3000"   // Development frontend
"https://13.201.120.175:3000"  // Development frontend (HTTPS)
```

### 2. **Nginx Reverse Proxy CORS**
```nginx
# CORS headers configured for:
add_header 'Access-Control-Allow-Origin' 'http://13.201.120.175' always;
add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD' always;
add_header 'Access-Control-Allow-Credentials' 'true' always;
```

### 3. **Frontend API Configuration**
```bash
# Production environment variables:
REACT_APP_API_URL=http://13.201.120.175/api
REACT_APP_ML_API_URL=http://13.201.120.175/api/ml
```

## 🚀 Deployment Commands

### Deploy Everything:
```bash
# Complete deployment with CORS
./deploy-to-ec2.sh 13.201.120.175 ~/.ssh/your-key.pem

# Deploy frontend with API configuration
./deploy-frontend-to-ec2.sh 13.201.120.175 ~/.ssh/your-key.pem

# Configure/test CORS separately
./configure-cors.sh 13.201.120.175 ~/.ssh/your-key.pem
```

## 🌐 Your Application URLs

After deployment, your application will be available at:

### **Frontend:**
- **Main App**: `http://13.201.120.175/`
- **CORS Test Page**: `http://13.201.120.175/cors-test.html`

### **Backend API:**
- **Base API**: `http://13.201.120.175/api/`
- **Health Check**: `http://13.201.120.175/health`
- **API Status**: `http://13.201.120.175/api/health/status`
- **Credentials Test**: `http://13.201.120.175/api/credentials/test`

### **H2 Database Console** (if enabled):
- **H2 Console**: `http://13.201.120.175/api/h2-console`

## 🔧 CORS Test Commands

### Test from Command Line:
```bash
# Test preflight OPTIONS request
curl -H "Origin: http://13.201.120.175" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type,Authorization" \
     -X OPTIONS \
     http://13.201.120.175/api/health/status

# Test actual API request
curl -H "Origin: http://13.201.120.175" \
     http://13.201.120.175/api/health/status

# Test health endpoint
curl http://13.201.120.175/health
```

### Expected CORS Headers:
```
Access-Control-Allow-Origin: http://13.201.120.175
Access-Control-Allow-Methods: GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD
Access-Control-Allow-Credentials: true
Access-Control-Allow-Headers: DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,Accept,Origin,Access-Control-Request-Method,Access-Control-Request-Headers
```

## 🛠️ Configuration Files Updated

### Backend:
- ✅ `src/main/java/com/smartshoe/config/CorsConfig.java` - Spring Boot CORS
- ✅ `src/main/java/com/smartshoe/config/WebSecurityConfig.java` - Security + CORS
- ✅ `src/main/java/com/smartshoe/config/PreflightCorsFilter.java` - OPTIONS handling

### Deployment Scripts:
- ✅ `deploy-to-ec2.sh` - Complete deployment with Nginx CORS
- ✅ `deploy-frontend-to-ec2.sh` - Frontend with API configuration
- ✅ `configure-cors.sh` - Dedicated CORS configuration tool

### Nginx Configuration:
- ✅ `/etc/nginx/conf.d/smartshoe-backend.conf` - Reverse proxy with CORS

## 🔍 Troubleshooting CORS Issues

### 1. **Check CORS Headers**
```bash
# Run the CORS configuration script
./configure-cors.sh

# Check what headers are being returned
curl -I -H "Origin: http://13.201.120.175" http://13.201.120.175/api/health/status
```

### 2. **Test in Browser**
- Open: `http://13.201.120.175/cors-test.html`
- Check browser console for any CORS errors
- All tests should show green success messages

### 3. **Check Service Status**
```bash
# SSH to your EC2 instance
ssh -i ~/.ssh/your-key.pem ec2-user@13.201.120.175

# Check backend service
sudo systemctl status smartshoe-backend

# Check nginx status
sudo systemctl status nginx

# View backend logs
sudo journalctl -u smartshoe-backend -f
```

### 4. **Common Issues & Solutions**

#### ❌ **"Access blocked by CORS policy"**
**Solution**: Run the CORS configuration script
```bash
./configure-cors.sh
```

#### ❌ **"No Access-Control-Allow-Origin header"**
**Solution**: Check Nginx configuration
```bash
ssh -i ~/.ssh/your-key.pem ec2-user@13.201.120.175
sudo nginx -t
sudo systemctl reload nginx
```

#### ❌ **"Preflight request doesn't pass"**
**Solution**: Verify OPTIONS handling
```bash
curl -X OPTIONS -H "Origin: http://13.201.120.175" http://13.201.120.175/api/health/status
```

## 🔒 Security Considerations

### Current CORS Configuration:
- ✅ **Specific Origin**: Only allows `http://13.201.120.175`
- ✅ **Credentials Enabled**: Supports authentication cookies/headers
- ✅ **All HTTP Methods**: GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD
- ✅ **Comprehensive Headers**: Supports all common request headers

### For Production Security:
1. **Enable HTTPS**: Configure SSL certificate for `https://13.201.120.175`
2. **Domain Name**: Use a proper domain instead of IP address
3. **Firewall**: Ensure only necessary ports are open
4. **Rate Limiting**: Consider adding rate limiting in Nginx

## 📊 Monitoring CORS

### Application Logs:
```bash
# Backend application logs
sudo journalctl -u smartshoe-backend -f

# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Nginx error logs
sudo tail -f /var/log/nginx/error.log
```

### Health Monitoring:
- **Backend Health**: `http://13.201.120.175/health`
- **API Status**: `http://13.201.120.175/api/health/status`
- **Frontend Test**: `http://13.201.120.175/cors-test.html`

## 🎉 Success Indicators

Your CORS configuration is working correctly when:

1. ✅ **Preflight requests return HTTP 204** with proper CORS headers
2. ✅ **API requests return HTTP 200** with `Access-Control-Allow-Origin` header
3. ✅ **Frontend can fetch from backend** without browser CORS errors
4. ✅ **CORS test page shows all green success messages**
5. ✅ **Browser console shows no CORS-related errors**

## 📞 Support Commands

### Quick Diagnostics:
```bash
# Run full CORS test
./configure-cors.sh 13.201.120.175 ~/.ssh/your-key.pem

# Check all endpoints
curl http://13.201.120.175/health
curl http://13.201.120.175/api/health/status
curl http://13.201.120.175/api/credentials/test

# View CORS test page
open http://13.201.120.175/cors-test.html
```

---

## 🎯 Summary

Your Smart Shoe application is now fully configured for CORS communication between:
- **Frontend**: `http://13.201.120.175/` 
- **Backend API**: `http://13.201.120.175/api/`

Both Spring Boot and Nginx are configured to allow cross-origin requests specifically for your public IP address `13.201.120.175`. 🚀