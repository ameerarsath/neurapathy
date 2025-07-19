import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.smartshoe.app',
  appName: 'Smart Shoe',
  webDir: 'dist',
  server: {
    androidScheme: 'https',
    allowNavigation: [
      'localhost:8080',
      'api.smartshoe.com',
      '*.smartshoe.com'
    ]
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
    Camera: {
      permissions: {
        camera: "This app needs access to camera to scan QR codes for device pairing",
        photos: "This app needs access to photos to save medical documentation"
      }
    },
    Device: {
      permissions: {
        device: "This app needs device information for diagnostics and support"
      }
    },
    Network: {
      permissions: {
        network: "This app needs network access to sync medical data"
      }
    }
  },
  ios: {
    scheme: "Smart Shoe",
    contentInset: "automatic"
  },
  android: {
    allowMixedContent: true,
    captureInput: true,
    webContentsDebuggingEnabled: false
  }
};

export default config;