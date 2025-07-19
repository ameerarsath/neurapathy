# 🔐 Smart Shoe API - Authentication Guide

## 🎯 Default Credentials

Your Smart Shoe API now has **4 default users** for testing:

| Username | Password   | Roles           | Description |
|----------|------------|-----------------|-------------|
| `admin`  | `admin123` | ADMIN, PROVIDER | Full access - can manage everything |
| `doctor` | `doctor123`| PROVIDER        | Healthcare provider access |
| `patient`| `patient123`| PATIENT        | Patient-level access |
| `demo`   | `demo`     | USER            | Basic demo access |

## 🌐 API Endpoints

### 📂 Public Endpoints (No Authentication Required)

Visit these URLs directly in your browser:

- **🏠 Health Check**: `http://localhost:8080/api/health`
- **📊 Status**: `http://localhost:8080/api/status`
- **🧪 Test**: `http://localhost:8080/api/test`
- **🔑 Credentials**: `http://localhost:8080/api/credentials`
- **📋 Endpoints List**: `http://localhost:8080/api/endpoints`
- **💾 H2 Database**: `http://localhost:8080/h2-console`
- **❤️ Actuator Health**: `http://localhost:8080/actuator/health`

### 🔒 Secured Endpoints (Authentication Required)

- **👥 Patients**: `http://localhost:8080/api/patients`
- **📱 Devices**: `http://localhost:8080/api/devices`
- **📋 Medical Readings**: `http://localhost:8080/api/medical-readings`

## 🔧 How to Access Secured Endpoints

### Option 1: Browser Authentication
1. Visit `http://localhost:8080/api/patients`
2. Browser will show login popup
3. Enter username: `admin` and password: `admin123`
4. Click OK

### Option 2: Postman/API Client
```
Method: GET
URL: http://localhost:8080/api/patients
Authorization: Basic Auth
Username: admin
Password: admin123
```

### Option 3: cURL Command
```bash
# Get all patients
curl -u admin:admin123 http://localhost:8080/api/patients

# Get application status  
curl -u admin:admin123 http://localhost:8080/api/status

# Test with doctor credentials
curl -u doctor:doctor123 http://localhost:8080/api/patients
```

### Option 4: JavaScript/Frontend
```javascript
// Using fetch with Basic Auth
const response = await fetch('http://localhost:8080/api/patients', {
  headers: {
    'Authorization': 'Basic ' + btoa('admin:admin123'),
    'Content-Type': 'application/json'
  }
});
```

## 🎨 Quick Testing URLs

**Start here** - Copy and paste these URLs in your browser:

1. **📊 Check API Status**: `http://localhost:8080/api/status`
2. **🔑 View All Credentials**: `http://localhost:8080/api/credentials`  
3. **📋 See All Endpoints**: `http://localhost:8080/api/endpoints`
4. **❤️ Health Check**: `http://localhost:8080/api/health`

## 🗄️ Database Access

**H2 Database Console** (No authentication required):
- **URL**: `http://localhost:8080/h2-console`
- **JDBC URL**: `jdbc:h2:mem:smartshoe`
- **Username**: `sa`
- **Password**: (leave blank)

## 🔐 Security Features

- ✅ **Basic Authentication** enabled
- ✅ **Role-based access control** (ADMIN, PROVIDER, PATIENT, USER)
- ✅ **Password encryption** using BCrypt
- ✅ **Public endpoints** for health checks and documentation
- ✅ **CSRF disabled** for API testing
- ✅ **H2 Console** accessible for development

## 🚨 Important Notes

1. **These are development credentials** - change them for production!
2. **Basic Auth** is used for simplicity - JWT can be implemented later
3. **All medical data endpoints** require authentication
4. **Public endpoints** are safe to call without credentials

## 🔄 Next Steps

1. **Test public endpoints** first to verify the API is working
2. **Use admin credentials** to access secured patient/device data
3. **Check the H2 database** to see stored data
4. **Implement frontend authentication** using these credentials

---

**🎉 Ready to Use!** Start with `http://localhost:8080/api/credentials` to see all available credentials.