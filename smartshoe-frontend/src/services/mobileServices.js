import { Capacitor } from '@capacitor/core';
import { Device } from '@capacitor/device';
import { Network } from '@capacitor/network';
import { PushNotifications } from '@capacitor/push-notifications';
import { LocalNotifications } from '@capacitor/local-notifications';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import { Haptics, ImpactStyle } from '@capacitor/haptics';
import { Preferences } from '@capacitor/preferences';

class MobileServices {
  constructor() {
    this.isNative = Capacitor.isNativePlatform();
    this.platform = Capacitor.getPlatform();
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    try {
      if (this.isNative) {
        await this.initializeNativeFeatures();
      }
      
      await this.initializeNetworkMonitoring();
      await this.initializeNotifications();
      
      this.initialized = true;
      console.log('Mobile services initialized successfully');
    } catch (error) {
      console.error('Failed to initialize mobile services:', error);
    }
  }

  async initializeNativeFeatures() {
    // Get device information
    this.deviceInfo = await Device.getInfo();
    this.deviceId = await Device.getId();
    
    console.log('Device Info:', this.deviceInfo);
    console.log('Device ID:', this.deviceId);

    // Initialize haptics
    if (this.deviceInfo.platform === 'ios' || this.deviceInfo.platform === 'android') {
      this.hapticsAvailable = true;
    }
  }

  async initializeNetworkMonitoring() {
    // Monitor network status
    this.networkStatus = await Network.getStatus();
    
    Network.addListener('networkStatusChange', (status) => {
      console.log('Network status changed:', status);
      this.handleNetworkChange(status);
    });
  }

  async initializeNotifications() {
    if (!this.isNative) return;

    try {
      // Request permissions
      const permissionStatus = await PushNotifications.requestPermissions();
      
      if (permissionStatus.receive === 'granted') {
        await PushNotifications.register();
        
        // Add listeners
        PushNotifications.addListener('registration', (token) => {
          console.log('Push registration success, token: ' + token.value);
          this.registerPushToken(token.value);
        });

        PushNotifications.addListener('registrationError', (error) => {
          console.error('Error on registration: ' + JSON.stringify(error));
        });

        PushNotifications.addListener('pushNotificationReceived', (notification) => {
          console.log('Push notification received: ', notification);
          this.handleIncomingNotification(notification);
        });

        PushNotifications.addListener('pushNotificationActionPerformed', (notification) => {
          console.log('Push notification action performed', notification);
          this.handleNotificationAction(notification);
        });
      }

      // Request local notification permissions
      await LocalNotifications.requestPermissions();
      
    } catch (error) {
      console.error('Failed to initialize notifications:', error);
    }
  }

  // Haptic Feedback
  async triggerHaptic(type = 'light') {
    if (!this.hapticsAvailable) return;

    try {
      const impactStyle = type === 'heavy' ? ImpactStyle.Heavy : 
                         type === 'medium' ? ImpactStyle.Medium : 
                         ImpactStyle.Light;
      
      await Haptics.impact({ style: impactStyle });
    } catch (error) {
      console.error('Haptic feedback failed:', error);
    }
  }

  // Camera Functions
  async takePicture(options = {}) {
    if (!this.isNative) {
      console.warn('Camera not available in web mode');
      return null;
    }

    try {
      const image = await Camera.getPhoto({
        quality: options.quality || 90,
        allowEditing: options.allowEditing || false,
        resultType: CameraResultType.DataUrl,
        source: options.source || CameraSource.Camera
      });

      return image;
    } catch (error) {
      console.error('Camera error:', error);
      throw error;
    }
  }

  // QR Code scanning for device pairing
  async scanQRCode() {
    try {
      const image = await this.takePicture({
        quality: 100,
        allowEditing: false,
        source: CameraSource.Camera
      });

      // Process QR code (would need additional QR library)
      return this.processQRCode(image);
    } catch (error) {
      console.error('QR scan failed:', error);
      throw error;
    }
  }

  // Local Storage
  async setData(key, value) {
    try {
      await Preferences.set({
        key,
        value: JSON.stringify(value)
      });
    } catch (error) {
      console.error('Failed to save data:', error);
    }
  }

  async getData(key) {
    try {
      const result = await Preferences.get({ key });
      return result.value ? JSON.parse(result.value) : null;
    } catch (error) {
      console.error('Failed to get data:', error);
      return null;
    }
  }

  async removeData(key) {
    try {
      await Preferences.remove({ key });
    } catch (error) {
      console.error('Failed to remove data:', error);
    }
  }

  // Notifications
  async scheduleLocalNotification(title, body, data = {}, scheduledTime = null) {
    try {
      const notification = {
        title,
        body,
        id: Date.now(),
        extra: data,
        iconColor: '#3B82F6',
        sound: 'beep.wav',
        group: 'smart-shoe-alerts'
      };

      if (scheduledTime) {
        notification.schedule = { at: new Date(scheduledTime) };
      }

      await LocalNotifications.schedule({
        notifications: [notification]
      });

      return notification.id;
    } catch (error) {
      console.error('Failed to schedule notification:', error);
    }
  }

  async sendCriticalAlert(message, data = {}) {
    await this.triggerHaptic('heavy');
    
    return await this.scheduleLocalNotification(
      '🚨 CRITICAL ALERT',
      message,
      { ...data, priority: 'critical' }
    );
  }

  async sendMedicationReminder(medicationName, scheduledTime) {
    return await this.scheduleLocalNotification(
      'Medication Reminder',
      `Time to take your ${medicationName}`,
      { type: 'medication', name: medicationName },
      scheduledTime
    );
  }

  async sendTestReminder(testType, scheduledTime) {
    return await this.scheduleLocalNotification(
      'Test Reminder',
      `Time for your ${testType} test`,
      { type: 'test', testType },
      scheduledTime
    );
  }

  // Network handling
  handleNetworkChange(status) {
    if (status.connected) {
      console.log('Network reconnected, syncing offline data...');
      this.syncOfflineData();
    } else {
      console.log('Network disconnected, enabling offline mode...');
      this.enableOfflineMode();
    }
  }

  async syncOfflineData() {
    try {
      const offlineQueue = await this.getData('offlineQueue') || [];
      
      for (const item of offlineQueue) {
        try {
          await this.sendToAPI(item.data, item.endpoint);
          // Remove from queue after successful sync
          const updatedQueue = offlineQueue.filter(q => q.id !== item.id);
          await this.setData('offlineQueue', updatedQueue);
        } catch (error) {
          console.error('Failed to sync item:', error);
          break; // Stop syncing if network fails
        }
      }
    } catch (error) {
      console.error('Failed to sync offline data:', error);
    }
  }

  async addToOfflineQueue(data, endpoint) {
    try {
      const offlineQueue = await this.getData('offlineQueue') || [];
      offlineQueue.push({
        id: Date.now(),
        data,
        endpoint,
        timestamp: new Date().toISOString()
      });
      
      await this.setData('offlineQueue', offlineQueue);
    } catch (error) {
      console.error('Failed to add to offline queue:', error);
    }
  }

  enableOfflineMode() {
    // Enable offline functionality
    document.body.classList.add('offline-mode');
    
    // Show offline indicator
    this.showOfflineIndicator();
  }

  showOfflineIndicator() {
    // Create offline indicator UI
    const indicator = document.createElement('div');
    indicator.id = 'offline-indicator';
    indicator.innerHTML = '📡 Offline Mode - Data will sync when connected';
    indicator.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      background: #F59E0B;
      color: white;
      text-align: center;
      padding: 8px;
      z-index: 9999;
      font-size: 14px;
    `;
    
    document.body.appendChild(indicator);
  }

  // API Communication
  async sendToAPI(data, endpoint) {
    // Implementation depends on your API structure
    const response = await fetch(`/api${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${await this.getData('authToken')}`
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    return response.json();
  }

  // Push notification handlers
  async registerPushToken(token) {
    try {
      await this.sendToAPI({
        token,
        platform: this.platform,
        deviceId: this.deviceId?.identifier,
        deviceInfo: this.deviceInfo
      }, '/notifications/register-token');
    } catch (error) {
      console.error('Failed to register push token:', error);
    }
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
      case 'medication_reminder':
        this.handleMedicationReminder(notification);
        break;
      default:
        console.log('Unknown notification type:', notification);
    }
  }

  handleNotificationAction(notification) {
    // Handle notification tap/action
    console.log('Notification action:', notification);
    
    // Navigate to relevant screen based on notification type
    if (notification.notification?.data?.type === 'critical_alert') {
      // Navigate to alerts page
      window.location.href = '/alerts';
    }
  }

  handleCriticalAlert(notification) {
    // Trigger strong haptic feedback
    this.triggerHaptic('heavy');
    
    // Show in-app alert
    this.showInAppAlert(notification.title, notification.body);
  }

  showInAppAlert(title, message) {
    // Create modal alert for critical notifications
    const modal = document.createElement('div');
    modal.innerHTML = `
      <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); z-index: 10000; display: flex; align-items: center; justify-content: center;">
        <div style="background: white; padding: 24px; border-radius: 12px; margin: 16px; max-width: 400px;">
          <h3 style="margin: 0 0 16px 0; color: #EF4444; font-size: 18px; font-weight: bold;">${title}</h3>
          <p style="margin: 0 0 24px 0; color: #374151;">${message}</p>
          <button onclick="this.parentElement.parentElement.remove()" style="background: #3B82F6; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer;">OK</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    // Auto-dismiss after 10 seconds
    setTimeout(() => {
      if (modal.parentElement) {
        modal.remove();
      }
    }, 10000);
  }

  // Battery optimization
  async getBatteryInfo() {
    if (!this.isNative) return null;

    try {
      const batteryInfo = await Device.getBatteryInfo();
      return batteryInfo;
    } catch (error) {
      console.error('Failed to get battery info:', error);
      return null;
    }
  }

  // Device orientation
  async getOrientation() {
    if (!this.isNative) return 'portrait';

    try {
      // This would require additional plugin
      return 'portrait'; // Default
    } catch (error) {
      console.error('Failed to get orientation:', error);
      return 'portrait';
    }
  }

  // Utility methods
  isOnline() {
    return this.networkStatus?.connected || navigator.onLine;
  }

  isPlatform(platform) {
    return this.platform === platform;
  }

  isIOS() {
    return this.isPlatform('ios');
  }

  isAndroid() {
    return this.isPlatform('android');
  }

  isWeb() {
    return this.isPlatform('web');
  }
}

// Export singleton instance
export const mobileServices = new MobileServices();

// Auto-initialize when imported
mobileServices.initialize();

export default mobileServices;