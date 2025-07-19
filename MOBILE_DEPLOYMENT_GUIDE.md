# 📱 Smart Shoe Mobile Deployment Guide

## Production-Ready Mobile App Development (IPA & APK)

This guide outlines the complete process to convert the Smart Shoe diabetic monitoring web application into production-ready mobile applications for iOS and Android.

## 🎯 Executive Summary

- **Current Status**: React 18.3.1 web application with Tailwind CSS
- **Recommended Approach**: Capacitor hybrid app (web-to-native)
- **Target Platforms**: iOS (App Store) + Android (Google Play)
- **Estimated Timeline**: 17-25 weeks for production deployment

## 📋 Prerequisites

### Development Environment
```bash
# Required Software
- Node.js 18+ (LTS recommended)
- Java 17+ (for Android builds)
- Xcode 15+ (for iOS builds, macOS only)
- Android Studio (for Android builds)
- Git

# Apple Developer Account ($99/year)
# Google Play Developer Account ($25 one-time)
```

## 🚀 Phase 1: Mobile Foundation Setup

### 1.1 Install Capacitor and Dependencies

```bash
cd /mnt/d/project/diabetic-smart-shoe/smartshoe-frontend

# Core Capacitor
npm install @capacitor/core @capacitor/cli

# Platform Support
npm install @capacitor/android @capacitor/ios

# Essential Mobile Plugins
npm install @capacitor/camera
npm install @capacitor/device
npm install @capacitor/bluetooth-le
npm install @capacitor/push-notifications
npm install @capacitor/local-notifications
npm install @capacitor/network
npm install @capacitor/storage
npm install @capacitor/haptics
npm install @capacitor/geolocation
npm install @capacitor/biometric-auth
npm install @capacitor/screen-orientation
npm install @capacitor/splash-screen
npm install @capacitor/status-bar
```

### 1.2 Initialize Capacitor

```bash
# Initialize Capacitor project
npx cap init smartshoe-app com.smartshoe.app

# Add mobile platforms
npx cap add android
npx cap add ios
```

### 1.3 Create PWA Manifest

Create `public/manifest.json`:
```json
{
  "name": "Smart Shoe - Diabetic Monitoring",
  "short_name": "SmartShoe",
  "description": "Professional diabetic neuropathy monitoring platform",
  "theme_color": "#3B82F6",
  "background_color": "#FFFFFF",
  "display": "standalone",
  "orientation": "portrait",
  "scope": "/",
  "start_url": "/",
  "categories": ["medical", "health"],
  "icons": [
    {
      "src": "icons/icon-72x72.png",
      "sizes": "72x72",
      "type": "image/png"
    },
    {
      "src": "icons/icon-96x96.png",
      "sizes": "96x96",
      "type": "image/png"
    },
    {
      "src": "icons/icon-128x128.png",
      "sizes": "128x128",
      "type": "image/png"
    },
    {
      "src": "icons/icon-144x144.png",
      "sizes": "144x144",
      "type": "image/png"
    },
    {
      "src": "icons/icon-152x152.png",
      "sizes": "152x152",
      "type": "image/png"
    },
    {
      "src": "icons/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "icons/icon-384x384.png",
      "sizes": "384x384",
      "type": "image/png"
    },
    {
      "src": "icons/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

## 🔧 Phase 2: Mobile Optimizations

### 2.1 Update Vite Configuration

Modify `vite.config.js`:
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\.smartshoe\.com\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
              },
              cacheKeyWillBeUsed: async ({ request }) => {
                return `${request.url}?version=1.0.0`
              }
            }
          }
        ]
      },
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg'],
      manifest: {
        name: 'Smart Shoe - Diabetic Monitoring',
        short_name: 'SmartShoe',
        description: 'Professional diabetic neuropathy monitoring platform',
        theme_color: '#3B82F6',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      }
    })
  ],
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'esbuild',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts'],
          ui: ['@headlessui/react', 'framer-motion']
        }
      }
    }
  },
  server: {
    port: 5173,
    host: true
  }
})
```

### 2.2 Create Capacitor Configuration

Create `capacitor.config.ts`:
```typescript
import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.smartshoe.app',
  appName: 'Smart Shoe',
  webDir: 'dist',
  server: {
    androidScheme: 'https'
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 3000,
      launchAutoHide: true,
      backgroundColor: "#3B82F6",
      androidSplashResourceName: "splash",
      androidScaleType: "CENTER_CROP",
      showSpinner: false,
      androidSpinnerStyle: "large",
      iosSpinnerStyle: "small",
      spinnerColor: "#FFFFFF",
      splashFullScreen: true,
      splashImmersive: true,
      layoutName: "launch_screen",
      useDialog: true,
    },
    PushNotifications: {
      presentationOptions: ["badge", "sound", "alert"],
    },
    LocalNotifications: {
      smallIcon: "ic_stat_icon_config_sample",
      iconColor: "#3B82F6",
      sound: "beep.wav",
    },
    BluetoothLe: {
      displayStrings: {
        scanning: "Scanning for smart shoe devices...",
        cancel: "Cancel",
        availableDevices: "Available Devices",
        noDeviceFound: "No devices found"
      }
    }
  },
  ios: {
    scheme: "Smart Shoe"
  },
  android: {
    allowMixedContent: true
  }
};

export default config;
```

## 📱 Phase 3: Native Features Implementation

### 3.1 Bluetooth Integration Service

Create `src/services/bluetoothService.js`:
```javascript
import { BleClient, numberToUUID } from '@capacitor-community/bluetooth-le';

class BluetoothService {
  constructor() {
    this.deviceId = null;
    this.serviceUUID = '6E400001-B5A3-F393-E0A9-E50E24DCCA9E';
    this.characteristicUUID = '6E400002-B5A3-F393-E0A9-E50E24DCCA9E';
    this.isConnected = false;
  }

  async initialize() {
    try {
      await BleClient.initialize();
      console.log('Bluetooth initialized');
      return true;
    } catch (error) {
      console.error('Bluetooth initialization failed:', error);
      return false;
    }
  }

  async scanForDevices() {
    try {
      await BleClient.requestLEScan(
        { services: [this.serviceUUID] },
        (result) => {
          console.log('Device found:', result);
          // Handle device discovery
          this.handleDeviceFound(result);
        }
      );

      setTimeout(async () => {
        await BleClient.stopLEScan();
      }, 10000);
    } catch (error) {
      console.error('Scan failed:', error);
    }
  }

  async connectToDevice(deviceId) {
    try {
      await BleClient.connect(deviceId);
      this.deviceId = deviceId;
      this.isConnected = true;
      
      // Start monitoring sensor data
      await this.startSensorDataMonitoring();
      
      return true;
    } catch (error) {
      console.error('Connection failed:', error);
      return false;
    }
  }

  async startSensorDataMonitoring() {
    try {
      await BleClient.startNotifications(
        this.deviceId,
        this.serviceUUID,
        this.characteristicUUID,
        (value) => {
          const sensorData = this.parseSensorData(value);
          this.handleSensorData(sensorData);
        }
      );
    } catch (error) {
      console.error('Failed to start notifications:', error);
    }
  }

  parseSensorData(value) {
    // Parse binary sensor data from smart shoe
    const dataView = new DataView(value.buffer);
    return {
      timestamp: Date.now(),
      pressure: {
        heel: dataView.getFloat32(0, true),
        midfoot: dataView.getFloat32(4, true),
        forefoot: dataView.getFloat32(8, true),
        toes: dataView.getFloat32(12, true)
      },
      temperature: dataView.getFloat32(16, true),
      humidity: dataView.getFloat32(20, true),
      batteryLevel: dataView.getUint8(24)
    };
  }

  handleSensorData(sensorData) {
    // Send to backend API
    this.sendToAPI(sensorData);
    
    // Update UI
    this.updateUI(sensorData);
    
    // Check for alerts
    this.checkForAlerts(sensorData);
  }

  async disconnect() {
    if (this.deviceId) {
      await BleClient.disconnect(this.deviceId);
      this.isConnected = false;
      this.deviceId = null;
    }
  }
}

export const bluetoothService = new BluetoothService();
```

### 3.2 Push Notifications Service

Create `src/services/notificationService.js`:
```javascript
import { PushNotifications } from '@capacitor/push-notifications';
import { LocalNotifications } from '@capacitor/local-notifications';

class NotificationService {
  async initialize() {
    // Request permission
    const permStatus = await PushNotifications.requestPermissions();
    
    if (permStatus.receive === 'granted') {
      // Register for push notifications
      await PushNotifications.register();
    }

    // Add listeners
    PushNotifications.addListener('registration', (token) => {
      console.log('Push registration success, token: ' + token.value);
      // Send token to backend
      this.registerTokenWithBackend(token.value);
    });

    PushNotifications.addListener('registrationError', (error) => {
      console.error('Error on registration: ' + JSON.stringify(error));
    });

    PushNotifications.addListener('pushNotificationReceived', (notification) => {
      console.log('Push notification received: ', notification);
      this.handleIncomingNotification(notification);
    });

    PushNotifications.addListener('pushNotificationActionPerformed', (notification) => {
      console.log('Push notification action performed', notification.actionId, notification.inputValue);
      this.handleNotificationAction(notification);
    });
  }

  async scheduleLocalNotification(title, body, data = {}) {
    await LocalNotifications.schedule({
      notifications: [
        {
          title: title,
          body: body,
          id: Date.now(),
          extra: data,
          iconColor: '#3B82F6',
          sound: 'beep.wav',
          attachments: [],
          actionTypeId: "",
          group: "smart-shoe-alerts"
        }
      ]
    });
  }

  async scheduleMedicationReminder(medicationName, time) {
    await LocalNotifications.schedule({
      notifications: [
        {
          title: 'Medication Reminder',
          body: `Time to take your ${medicationName}`,
          id: Date.now(),
          schedule: { at: new Date(time) },
          sound: 'medication-alert.wav',
          extra: { type: 'medication', name: medicationName }
        }
      ]
    });
  }

  async scheduleTestReminder(testType, scheduledTime) {
    await LocalNotifications.schedule({
      notifications: [
        {
          title: 'Test Reminder',
          body: `Time for your ${testType} test`,
          id: Date.now(),
          schedule: { at: new Date(scheduledTime) },
          sound: 'test-reminder.wav',
          extra: { type: 'test', testType }
        }
      ]
    });
  }

  async sendCriticalAlert(alertData) {
    // For critical medical alerts
    await LocalNotifications.schedule({
      notifications: [
        {
          title: '🚨 CRITICAL ALERT',
          body: alertData.message,
          id: Date.now(),
          sound: 'critical-alert.wav',
          iconColor: '#EF4444',
          extra: { ...alertData, priority: 'critical' },
          ongoing: true // Keep notification until user interaction
        }
      ]
    });
  }

  handleIncomingNotification(notification) {
    // Handle different types of notifications
    switch (notification.data?.type) {
      case 'critical_alert':
        this.handleCriticalAlert(notification);
        break;
      case 'test_result':
        this.handleTestResult(notification);
        break;
      case 'device_status':
        this.handleDeviceStatus(notification);
        break;
      default:
        console.log('Unknown notification type');
    }
  }

  async registerTokenWithBackend(token) {
    try {
      await fetch('/api/notifications/register-token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          token,
          platform: 'mobile',
          deviceType: 'smart-shoe-app'
        })
      });
    } catch (error) {
      console.error('Failed to register token:', error);
    }
  }
}

export const notificationService = new NotificationService();
```

## 🔧 Phase 4: Backend Mobile API Optimizations

### 4.1 Mobile-Specific API Endpoints

Add to `MLPredictionController.java`:
```java
@PostMapping("/mobile/predict/batch")
@Operation(summary = "Batch prediction for mobile", description = "Optimized batch predictions for mobile devices")
public ResponseEntity<Map<String, Object>> mobileBatchPredict(
        @RequestBody MobileBatchPredictionRequest request,
        HttpServletRequest httpRequest) {
    
    try {
        // Mobile-optimized processing
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("predictions", processMobileBatch(request));
        response.put("timestamp", LocalDateTime.now());
        response.put("cached", true); // Mobile apps prefer cached responses
        
        return ResponseEntity.ok(response);
        
    } catch (Exception e) {
        return ResponseEntity.badRequest()
            .body(Map.of("success", false, "error", e.getMessage()));
    }
}

@GetMapping("/mobile/health")
@Operation(summary = "Mobile health check", description = "Lightweight health check for mobile")
public ResponseEntity<Map<String, Object>> mobileHealthCheck() {
    Map<String, Object> health = new HashMap<>();
    health.put("status", "healthy");
    health.put("timestamp", LocalDateTime.now());
    health.put("version", "3.0.0");
    health.put("features", Arrays.asList("bluetooth", "notifications", "offline"));
    
    return ResponseEntity.ok(health);
}
```

### 4.2 Mobile CORS Configuration

Update `SecurityConfig.java`:
```java
@Bean
public CorsConfigurationSource corsConfigurationSource() {
    CorsConfiguration configuration = new CorsConfiguration();
    
    // Mobile app origins
    configuration.setAllowedOriginPatterns(Arrays.asList(
        "capacitor://localhost",
        "ionic://localhost",
        "http://localhost:*",
        "https://localhost:*",
        "https://app.smartshoe.com",
        "https://*.smartshoe.com"
    ));
    
    configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"));
    configuration.setAllowedHeaders(Arrays.asList("*"));
    configuration.setAllowCredentials(true);
    configuration.setMaxAge(3600L);
    
    UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
    source.registerCorsConfiguration("/**", configuration);
    return source;
}
```

## 📦 Phase 5: Build and Deployment

### 5.1 Build Commands

```bash
# Build web assets
npm run build

# Sync with Capacitor
npx cap sync

# Open in IDE for final build
npx cap open android  # For Android Studio
npx cap open ios      # For Xcode
```

### 5.2 Android Build Configuration

Create `android/app/build.gradle` modifications:
```gradle
android {
    compileSdkVersion 34
    
    defaultConfig {
        applicationId "com.smartshoe.app"
        minSdkVersion 26
        targetSdkVersion 34
        versionCode 1
        versionName "1.0.0"
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }
    
    buildTypes {
        release {
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
            signingConfig signingConfigs.release
        }
    }
}
```

### 5.3 iOS Build Configuration

Update `ios/App/Info.plist`:
```xml
<key>NSBluetoothAlwaysUsageDescription</key>
<string>This app needs Bluetooth to connect to your Smart Shoe device for medical monitoring.</string>
<key>NSBluetoothPeripheralUsageDescription</key>
<string>This app needs Bluetooth to connect to your Smart Shoe device for medical monitoring.</string>
<key>NSLocationWhenInUseUsageDescription</key>
<string>This app needs location access for Bluetooth device discovery.</string>
<key>NSCameraUsageDescription</key>
<string>This app needs camera access to scan QR codes for device pairing.</string>
```

## 🏥 Phase 6: Medical Compliance & Security

### 6.1 HIPAA Compliance Checklist

- ✅ End-to-end encryption for all medical data
- ✅ Secure authentication with biometric options
- ✅ Audit logging for all data access
- ✅ Data minimization and retention policies
- ✅ Patient consent management
- ✅ Secure data transmission (TLS 1.3)
- ✅ Local data encryption on device
- ✅ Remote wipe capabilities

### 6.2 App Store Compliance

**iOS App Store:**
- Medical device registration documentation
- Privacy policy and terms of service
- HIPAA compliance statement
- Clinical validation reports
- User safety documentation

**Google Play Store:**
- Medical app declaration
- Data safety section completion
- Privacy policy compliance
- Medical device certification
- Target API level compliance

## 📊 Phase 7: Testing & Quality Assurance

### 7.1 Testing Strategy

```bash
# Unit Testing
npm test

# E2E Testing with Capacitor
npm install @capacitor/testing
npx cap test

# Device Testing
# - Test on real iOS/Android devices
# - Verify Bluetooth connectivity
# - Test offline functionality
# - Validate push notifications
# - Performance testing under load
```

### 7.2 Performance Monitoring

```javascript
// Add to main.jsx
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

// Monitor Core Web Vitals
getCLS(console.log);
getFID(console.log);
getFCP(console.log);
getLCP(console.log);
getTTFB(console.log);
```

## 🚀 Phase 8: Deployment Pipeline

### 8.1 Automated Build Pipeline

Create `.github/workflows/mobile-build.yml`:
```yaml
name: Mobile Build Pipeline

on:
  push:
    branches: [main, release/*]
  pull_request:
    branches: [main]

jobs:
  build-android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm ci
      - name: Build web assets
        run: npm run build
      - name: Setup Java
        uses: actions/setup-java@v3
        with:
          java-version: '17'
          distribution: 'temurin'
      - name: Build Android APK
        run: |
          npx cap sync android
          cd android && ./gradlew assembleRelease
      - name: Upload APK
        uses: actions/upload-artifact@v3
        with:
          name: app-release.apk
          path: android/app/build/outputs/apk/release/

  build-ios:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm ci
      - name: Build web assets
        run: npm run build
      - name: Setup iOS
        run: |
          npx cap sync ios
          cd ios && xcodebuild -workspace App.xcworkspace -scheme App -configuration Release -destination generic/platform=iOS -archivePath App.xcarchive archive
```

## 📋 Final Deployment Checklist

### Pre-Launch
- [ ] All medical compliance requirements met
- [ ] Security audit completed
- [ ] Performance testing passed
- [ ] Device testing on target hardware
- [ ] App store assets prepared (screenshots, descriptions)
- [ ] Privacy policy and terms updated
- [ ] Beta testing completed

### Launch
- [ ] Submit to App Store Connect (iOS)
- [ ] Submit to Google Play Console (Android)
- [ ] Monitor crash reporting
- [ ] Set up analytics and monitoring
- [ ] Prepare customer support documentation
- [ ] Monitor user feedback and ratings

## 📈 Post-Launch Monitoring

### Key Metrics to Monitor
- App crash rates (target: <1%)
- Bluetooth connection success rate (target: >98%)
- API response times (target: <500ms)
- User engagement metrics
- Medical data accuracy validation
- Battery usage optimization

## 🔧 Troubleshooting Common Issues

### Bluetooth Connection Issues
```javascript
// Add connection retry logic
async connectWithRetry(deviceId, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await this.connectToDevice(deviceId);
      return true;
    } catch (error) {
      console.log(`Connection attempt ${i + 1} failed:`, error);
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
}
```

### Offline Data Sync
```javascript
// Implement offline queue
class OfflineDataSync {
  constructor() {
    this.queue = [];
    this.isOnline = navigator.onLine;
    this.setupNetworkListeners();
  }
  
  setupNetworkListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.processQueue();
    });
    
    window.addEventListener('offline', () => {
      this.isOnline = false;
    });
  }
  
  async addToQueue(data) {
    this.queue.push({ ...data, timestamp: Date.now() });
    if (this.isOnline) {
      await this.processQueue();
    }
  }
  
  async processQueue() {
    while (this.queue.length > 0 && this.isOnline) {
      const item = this.queue.shift();
      try {
        await this.sendToAPI(item);
      } catch (error) {
        // Re-queue if network error
        this.queue.unshift(item);
        break;
      }
    }
  }
}
```

---

## 📞 Support & Resources

- **Technical Documentation**: Internal development team
- **Medical Compliance**: Healthcare regulatory consultant
- **App Store Support**: Apple Developer Support / Google Play Support
- **Security Audit**: Third-party security assessment firm

**Estimated Total Cost**: $50,000 - $75,000 (including development, compliance, and first-year operational costs)

**Timeline**: 17-25 weeks from project start to app store approval