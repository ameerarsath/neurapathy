# CORS Configuration Setup

This document explains how to resolve CORS (Cross-Origin Resource Sharing) errors in the Smart Shoe backend application.

## Files Created

1. **`src/main/java/com/smartshoe/config/CorsConfig.java`** - Main CORS configuration
2. **`src/main/java/com/smartshoe/config/WebSecurityConfig.java`** - Security configuration with CORS integration
3. **`src/main/java/com/smartshoe/config/PreflightCorsFilter.java`** - Filter for handling preflight OPTIONS requests
4. **`src/main/resources/cors.properties`** - External CORS configuration properties

## Error Resolution

The CORS error you encountered:
```
Access to XMLHttpRequest at 'http://13.201.120.175:8080/api/auth/login' from origin 'http://13.201.120.175' has been blocked by CORS policy
```

Is resolved by these configurations which allow:
- Frontend origin: `http://13.201.120.175`
- Backend API: `http://13.201.120.175:8080`

## Production Deployment Steps

### 1. Add Files to Your Backend Project

Copy the created configuration files to your backend project:

```bash
# Navigate to your backend project
cd /path/to/your/backend/project

# Create the config directory if it doesn't exist
mkdir -p src/main/java/com/smartshoe/config

# Copy the configuration files
cp /mnt/d/project/diabetic-smart-shoe/backend/src/main/java/com/smartshoe/config/* src/main/java/com/smartshoe/config/
cp /mnt/d/project/diabetic-smart-shoe/backend/src/main/resources/cors.properties src/main/resources/
```

### 2. Update Package Names (if different)

If your backend uses a different package structure, update the package declarations in:
- `CorsConfig.java`
- `WebSecurityConfig.java` 
- `PreflightCorsFilter.java`

Change `package com.smartshoe.config;` to match your project structure.

### 3. Add Required Dependencies

Ensure your `pom.xml` includes:

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
```

### 4. Build and Deploy

```bash
# Clean and build the project
mvn clean package

# Copy JAR to server
scp target/your-app.jar ubuntu@13.201.120.175:/path/to/deployment/

# SSH to server and restart application
ssh ubuntu@13.201.120.175
sudo systemctl restart your-spring-boot-service
```

### 5. Verify CORS Headers

Test that CORS headers are properly set:

```bash
# Test preflight request
curl -H "Origin: http://13.201.120.175" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type,Authorization" \
     -X OPTIONS \
     http://13.201.120.175:8080/api/auth/login

# Expected response should include:
# Access-Control-Allow-Origin: http://13.201.120.175
# Access-Control-Allow-Methods: GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD
# Access-Control-Allow-Headers: ...
# Access-Control-Allow-Credentials: true
```

## Configuration Details

### Allowed Origins
- **Production**: `http://13.201.120.175`, `https://13.201.120.175`
- **Development**: `http://localhost:3000`, `http://localhost:5173`
- **Mobile**: `capacitor://localhost`, `ionic://localhost`

### Allowed Methods
- GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD

### Security Features
- Credentials support enabled
- Preflight request handling
- Custom headers support
- 1-hour preflight cache

## Troubleshooting

### If CORS errors persist:

1. **Check application logs** for any configuration errors
2. **Verify package names** match your project structure
3. **Test without authentication** first on public endpoints
4. **Clear browser cache** and hard refresh
5. **Check network tab** in browser dev tools for actual response headers

### Alternative Quick Fix (Nginx)

If you're using Nginx as a reverse proxy, you can also add CORS headers there:

```nginx
location /api/ {
    proxy_pass http://localhost:8080;
    
    add_header 'Access-Control-Allow-Origin' 'http://13.201.120.175' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
    add_header 'Access-Control-Allow-Credentials' 'true' always;
    
    if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Max-Age' 1728000;
        add_header 'Content-Type' 'text/plain; charset=utf-8';
        add_header 'Content-Length' 0;
        return 204;
    }
}
```

## Production Security Notes

For production environments, consider:
1. Using HTTPS instead of HTTP
2. Restricting origins to only your domain
3. Implementing proper authentication and authorization
4. Regular security audits of CORS configuration

## Support

If you continue to experience CORS issues after implementing these configurations, check:
- Backend application logs
- Network requests in browser developer tools
- Server response headers
- Firewall and security group settings