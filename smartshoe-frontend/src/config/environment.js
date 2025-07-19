/**
 * Environment Configuration for Smart Shoe Application
 * Centralizes all environment variables and provides type-safe access
 */

// Environment detection
export const ENV = {
  NODE_ENV: import.meta.env.MODE,
  isDevelopment: import.meta.env.DEV,
  isProduction: import.meta.env.PROD,
  isTesting: import.meta.env.MODE === 'test'
}

// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8080',
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080/api',
  ML_API_URL: import.meta.env.VITE_ML_API_URL || 'http://localhost:8080/api/ml',
  WEBSOCKET_URL: import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8080/ws',
  TIMEOUT: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000'),
  RETRY_ATTEMPTS: parseInt(import.meta.env.VITE_REQUEST_RETRY_ATTEMPTS || '3'),
  SESSION_TIMEOUT: parseInt(import.meta.env.VITE_SESSION_TIMEOUT || '1800000'),
  TOKEN_REFRESH_THRESHOLD: parseInt(import.meta.env.VITE_TOKEN_REFRESH_THRESHOLD || '300000')
}

// Application Configuration
export const APP_CONFIG = {
  NAME: import.meta.env.VITE_APP_NAME || 'Smart Shoe',
  VERSION: import.meta.env.VITE_APP_VERSION || '1.0.0',
  ENVIRONMENT: import.meta.env.VITE_APP_ENVIRONMENT || 'development',
  DEBUG: import.meta.env.VITE_APP_DEBUG === 'true',
  BUILD_TARGET: import.meta.env.VITE_BUILD_TARGET || 'development'
}

// Mobile Configuration
export const MOBILE_CONFIG = {
  APP_ID: import.meta.env.VITE_MOBILE_APP_ID || 'com.smartshoe.app',
  APP_NAME: import.meta.env.VITE_MOBILE_APP_NAME || 'Smart Shoe',
  DEEP_LINK_SCHEME: import.meta.env.VITE_MOBILE_DEEP_LINK_SCHEME || 'smartshoe'
}

// Feature Flags
export const FEATURES = {
  BLUETOOTH: import.meta.env.VITE_ENABLE_BLUETOOTH === 'true',
  PUSH_NOTIFICATIONS: import.meta.env.VITE_ENABLE_PUSH_NOTIFICATIONS === 'true',
  CAMERA: import.meta.env.VITE_ENABLE_CAMERA === 'true',
  BIOMETRIC_AUTH: import.meta.env.VITE_ENABLE_BIOMETRIC_AUTH === 'true',
  OFFLINE_MODE: import.meta.env.VITE_ENABLE_OFFLINE_MODE === 'true',
  ANALYTICS: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
  CRASH_REPORTING: import.meta.env.VITE_ENABLE_CRASH_REPORTING === 'true',
  SERVICE_WORKER: import.meta.env.VITE_ENABLE_SERVICE_WORKER === 'true',
  DARK_MODE: import.meta.env.VITE_ENABLE_DARK_MODE === 'true',
  HAPTIC_FEEDBACK: import.meta.env.VITE_ENABLE_HAPTIC_FEEDBACK === 'true'
}

// Security Configuration
export const SECURITY_CONFIG = {
  HTTPS_REDIRECT: import.meta.env.VITE_ENABLE_HTTPS_REDIRECT === 'true',
  CSP_ENABLED: import.meta.env.VITE_CSP_ENABLED === 'true',
  ALLOWED_HOSTS: (import.meta.env.VITE_ALLOWED_HOSTS || 'localhost').split(','),
  ALLOWED_ORIGINS: (import.meta.env.VITE_ALLOWED_ORIGINS || 'http://localhost:8080').split(',')
}

// Medical Compliance
export const COMPLIANCE_CONFIG = {
  HIPAA_COMPLIANT: import.meta.env.VITE_HIPAA_COMPLIANT === 'true',
  AUDIT_LOGGING: import.meta.env.VITE_ENABLE_AUDIT_LOGGING === 'true',
  DATA_ENCRYPTION: import.meta.env.VITE_DATA_ENCRYPTION_ENABLED === 'true',
  BIOMETRIC_AUTH_REQUIRED: import.meta.env.VITE_BIOMETRIC_AUTH_REQUIRED === 'true',
  SESSION_RECORDING_DISABLED: import.meta.env.VITE_SESSION_RECORDING_DISABLED === 'true'
}

// Third-party Services
export const SERVICES_CONFIG = {
  FIREBASE: {
    API_KEY: import.meta.env.VITE_FIREBASE_API_KEY,
    PROJECT_ID: import.meta.env.VITE_FIREBASE_PROJECT_ID,
    MESSAGING_SENDER_ID: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
    APP_ID: import.meta.env.VITE_FIREBASE_APP_ID
  },
  ANALYTICS: {
    GOOGLE_ANALYTICS_ID: import.meta.env.VITE_GOOGLE_ANALYTICS_ID,
    MIXPANEL_TOKEN: import.meta.env.VITE_MIXPANEL_TOKEN
  },
  ERROR_TRACKING: {
    SENTRY_DSN: import.meta.env.VITE_SENTRY_DSN,
    SENTRY_ENVIRONMENT: import.meta.env.VITE_SENTRY_ENVIRONMENT
  }
}

// Performance Configuration
export const PERFORMANCE_CONFIG = {
  CACHE_STRATEGY: import.meta.env.VITE_CACHE_STRATEGY || 'cache-first',
  OFFLINE_CACHE_DURATION: parseInt(import.meta.env.VITE_OFFLINE_CACHE_DURATION || '86400000'),
  API_CACHE_DURATION: parseInt(import.meta.env.VITE_API_CACHE_DURATION || '300000'),
  ANIMATION_DURATION: parseInt(import.meta.env.VITE_ANIMATION_DURATION || '300')
}

// UI Configuration
export const UI_CONFIG = {
  THEME_MODE: import.meta.env.VITE_THEME_MODE || 'light',
  DEFAULT_LANGUAGE: import.meta.env.VITE_DEFAULT_LANGUAGE || 'en',
  SUPPORTED_LANGUAGES: (import.meta.env.VITE_SUPPORTED_LANGUAGES || 'en').split(',')
}

// Development Configuration
export const DEV_CONFIG = {
  REDUX_DEVTOOLS: import.meta.env.VITE_ENABLE_REDUX_DEVTOOLS === 'true',
  CONSOLE_LOGS: import.meta.env.VITE_ENABLE_CONSOLE_LOGS === 'true',
  PERFORMANCE_MONITORING: import.meta.env.VITE_ENABLE_PERFORMANCE_MONITORING === 'true',
  ERROR_BOUNDARY: import.meta.env.VITE_ENABLE_ERROR_BOUNDARY === 'true',
  BUNDLE_ANALYZER: import.meta.env.VITE_BUNDLE_ANALYZER === 'true',
  SOURCE_MAPS: import.meta.env.VITE_SOURCE_MAPS === 'true'
}

// Validation function to check required environment variables
export const validateEnvironment = () => {
  const required = [
    'VITE_API_URL',
    'VITE_API_BASE_URL'
  ]

  const missing = required.filter(key => !import.meta.env[key])
  
  if (missing.length > 0) {
    console.error('Missing required environment variables:', missing)
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`)
  }

  // Log configuration in development
  if (ENV.isDevelopment && DEV_CONFIG.CONSOLE_LOGS) {
    console.log('🔧 Environment Configuration:', {
      ENV,
      API_CONFIG,
      APP_CONFIG,
      FEATURES,
      MOBILE_CONFIG
    })
  }
}

// Helper functions
export const getApiUrl = (endpoint = '') => {
  const baseUrl = API_CONFIG.API_BASE_URL.replace(/\/$/, '')
  const cleanEndpoint = endpoint.replace(/^\//, '')
  return cleanEndpoint ? `${baseUrl}/${cleanEndpoint}` : baseUrl
}

export const getMLApiUrl = (endpoint = '') => {
  const baseUrl = API_CONFIG.ML_API_URL.replace(/\/$/, '')
  const cleanEndpoint = endpoint.replace(/^\//, '')
  return cleanEndpoint ? `${baseUrl}/${cleanEndpoint}` : baseUrl
}

export const getWebSocketUrl = () => {
  return API_CONFIG.WEBSOCKET_URL
}

export const isFeatureEnabled = (feature) => {
  return FEATURES[feature] === true
}

export const isMobileEnvironment = () => {
  return window.Capacitor?.isNativePlatform() || false
}

export const getPlatform = () => {
  if (typeof window !== 'undefined' && window.Capacitor) {
    return window.Capacitor.getPlatform()
  }
  return 'web'
}

// Export everything as default for easy importing
export default {
  ENV,
  API_CONFIG,
  APP_CONFIG,
  MOBILE_CONFIG,
  FEATURES,
  SECURITY_CONFIG,
  COMPLIANCE_CONFIG,
  SERVICES_CONFIG,
  PERFORMANCE_CONFIG,
  UI_CONFIG,
  DEV_CONFIG,
  validateEnvironment,
  getApiUrl,
  getMLApiUrl,
  getWebSocketUrl,
  isFeatureEnabled,
  isMobileEnvironment,
  getPlatform
}