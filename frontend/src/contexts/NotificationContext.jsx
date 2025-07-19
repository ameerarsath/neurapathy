import React, { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { api } from '@services/api'

// Create context
const NotificationContext = createContext()

// Custom hook to use notifications
export const useNotifications = () => {
  const context = useContext(NotificationContext)
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider')
  }
  return context
}

// Notification types
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info',
  MEDICAL_ALERT: 'medical_alert',
  DEVICE_ALERT: 'device_alert',
  SYSTEM_ALERT: 'system_alert'
}

// Notification priorities
export const NOTIFICATION_PRIORITIES = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical'
}

// Notification provider component
export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([])
  const [preferences, setPreferences] = useState({
    enabled: true,
    sound: true,
    vibration: true,
    desktop: true,
    email: false,
    sms: false,
    push: true,
    types: {
      [NOTIFICATION_TYPES.SUCCESS]: true,
      [NOTIFICATION_TYPES.ERROR]: true,
      [NOTIFICATION_TYPES.WARNING]: true,
      [NOTIFICATION_TYPES.INFO]: true,
      [NOTIFICATION_TYPES.MEDICAL_ALERT]: true,
      [NOTIFICATION_TYPES.DEVICE_ALERT]: true,
      [NOTIFICATION_TYPES.SYSTEM_ALERT]: true
    },
    priorities: {
      [NOTIFICATION_PRIORITIES.LOW]: true,
      [NOTIFICATION_PRIORITIES.MEDIUM]: true,
      [NOTIFICATION_PRIORITIES.HIGH]: true,
      [NOTIFICATION_PRIORITIES.CRITICAL]: true
    }
  })
  const [unreadCount, setUnreadCount] = useState(0)

  // Load preferences from localStorage
  useEffect(() => {
    const savedPreferences = localStorage.getItem('notification_preferences')
    if (savedPreferences) {
      try {
        setPreferences(JSON.parse(savedPreferences))
      } catch (error) {
        console.error('Error loading notification preferences:', error)
      }
    }
  }, [])

  // Save preferences to localStorage
  useEffect(() => {
    localStorage.setItem('notification_preferences', JSON.stringify(preferences))
  }, [preferences])

  // Request notification permission
  useEffect(() => {
    if (preferences.desktop && 'Notification' in window) {
      if (Notification.permission === 'default') {
        Notification.requestPermission()
      }
    }
  }, [preferences.desktop])

  // Calculate unread count
  useEffect(() => {
    const unread = notifications.filter(n => !n.read).length
    setUnreadCount(unread)
  }, [notifications])

  // Add notification
  const addNotification = useCallback((notification) => {
    const newNotification = {
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      read: false,
      ...notification
    }

    setNotifications(prev => [newNotification, ...prev])

    // Show appropriate notification based on type and preferences
    if (preferences.enabled && preferences.types[notification.type]) {
      showNotification(newNotification)
    }

    return newNotification.id
  }, [preferences])

  // Show notification
  const showNotification = useCallback((notification) => {
    const { type, title, message, priority, duration = 4000 } = notification

    // Show toast notification
    const toastOptions = {
      duration,
      position: 'top-right',
      style: getNotificationStyle(type, priority)
    }

    switch (type) {
      case NOTIFICATION_TYPES.SUCCESS:
        toast.success(title || message, toastOptions)
        break
      case NOTIFICATION_TYPES.ERROR:
        toast.error(title || message, toastOptions)
        break
      case NOTIFICATION_TYPES.WARNING:
        toast.error(title || message, toastOptions)
        break
      case NOTIFICATION_TYPES.MEDICAL_ALERT:
        toast.error(`🏥 ${title || message}`, { ...toastOptions, duration: 8000 })
        break
      case NOTIFICATION_TYPES.DEVICE_ALERT:
        toast.error(`⚠️ ${title || message}`, { ...toastOptions, duration: 6000 })
        break
      case NOTIFICATION_TYPES.SYSTEM_ALERT:
        toast.error(`🔧 ${title || message}`, { ...toastOptions, duration: 6000 })
        break
      default:
        toast(title || message, toastOptions)
    }

    // Show desktop notification
    if (preferences.desktop && 'Notification' in window && Notification.permission === 'granted') {
      new Notification(title || 'Smart Shoe Alert', {
        body: message,
        icon: '/favicon.ico',
        tag: notification.id,
        renotify: priority === NOTIFICATION_PRIORITIES.CRITICAL
      })
    }

    // Play sound
    if (preferences.sound && priority !== NOTIFICATION_PRIORITIES.LOW) {
      playNotificationSound(type, priority)
    }

    // Vibrate
    if (preferences.vibration && 'vibrate' in navigator) {
      const vibrationPattern = getVibrationPattern(priority)
      navigator.vibrate(vibrationPattern)
    }
  }, [preferences])

  // Get notification style based on type and priority
  const getNotificationStyle = (type, priority) => {
    const baseStyle = {
      borderRadius: '8px',
      padding: '16px',
      fontSize: '14px',
      fontWeight: priority === NOTIFICATION_PRIORITIES.CRITICAL ? 'bold' : 'normal'
    }

    switch (type) {
      case NOTIFICATION_TYPES.SUCCESS:
        return { ...baseStyle, background: '#10B981', color: '#ffffff' }
      case NOTIFICATION_TYPES.ERROR:
      case NOTIFICATION_TYPES.MEDICAL_ALERT:
        return { ...baseStyle, background: '#EF4444', color: '#ffffff' }
      case NOTIFICATION_TYPES.WARNING:
      case NOTIFICATION_TYPES.DEVICE_ALERT:
        return { ...baseStyle, background: '#F59E0B', color: '#ffffff' }
      case NOTIFICATION_TYPES.SYSTEM_ALERT:
        return { ...baseStyle, background: '#6B7280', color: '#ffffff' }
      default:
        return { ...baseStyle, background: '#3B82F6', color: '#ffffff' }
    }
  }

  // Play notification sound
  const playNotificationSound = (type, priority) => {
    try {
      let soundFile = '/sounds/notification.mp3'
      
      if (type === NOTIFICATION_TYPES.MEDICAL_ALERT) {
        soundFile = '/sounds/medical-alert.mp3'
      } else if (priority === NOTIFICATION_PRIORITIES.CRITICAL) {
        soundFile = '/sounds/critical-alert.mp3'
      }

      const audio = new Audio(soundFile)
      audio.volume = 0.3
      audio.play().catch(() => {
        // Fallback to default system sound
        console.warn('Could not play notification sound')
      })
    } catch (error) {
      console.error('Error playing notification sound:', error)
    }
  }

  // Get vibration pattern based on priority
  const getVibrationPattern = (priority) => {
    switch (priority) {
      case NOTIFICATION_PRIORITIES.CRITICAL:
        return [200, 100, 200, 100, 200]
      case NOTIFICATION_PRIORITIES.HIGH:
        return [200, 100, 200]
      case NOTIFICATION_PRIORITIES.MEDIUM:
        return [200]
      default:
        return [100]
    }
  }

  // Mark notification as read
  const markAsRead = useCallback((notificationId) => {
    setNotifications(prev => 
      prev.map(notification => 
        notification.id === notificationId 
          ? { ...notification, read: true }
          : notification
      )
    )
  }, [])

  // Mark all notifications as read
  const markAllAsRead = useCallback(() => {
    setNotifications(prev => 
      prev.map(notification => ({ ...notification, read: true }))
    )
  }, [])

  // Remove notification
  const removeNotification = useCallback((notificationId) => {
    setNotifications(prev => prev.filter(n => n.id !== notificationId))
  }, [])

  // Clear all notifications
  const clearAllNotifications = useCallback(() => {
    setNotifications([])
  }, [])

  // Update preferences
  const updatePreferences = useCallback((newPreferences) => {
    setPreferences(prev => ({ ...prev, ...newPreferences }))
  }, [])

  // Load notifications from server
  const loadNotifications = useCallback(async () => {
    try {
      const response = await api.alert.getAlerts({ limit: 50 })
      const serverNotifications = response.data.map(alert => ({
        id: alert.id,
        type: alert.type,
        title: alert.title,
        message: alert.message,
        priority: alert.priority,
        timestamp: alert.createdAt,
        read: alert.acknowledged,
        data: alert.data
      }))
      setNotifications(serverNotifications)
    } catch (error) {
      console.error('Error loading notifications:', error)
    }
  }, [])

  // Sync with server
  useEffect(() => {
    loadNotifications()
    const interval = setInterval(loadNotifications, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [loadNotifications])

  // Helper functions for common notification types
  const showSuccess = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.SUCCESS,
      message,
      priority: NOTIFICATION_PRIORITIES.LOW,
      ...options
    })
  }, [addNotification])

  const showError = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.ERROR,
      message,
      priority: NOTIFICATION_PRIORITIES.HIGH,
      ...options
    })
  }, [addNotification])

  const showWarning = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.WARNING,
      message,
      priority: NOTIFICATION_PRIORITIES.MEDIUM,
      ...options
    })
  }, [addNotification])

  const showInfo = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.INFO,
      message,
      priority: NOTIFICATION_PRIORITIES.LOW,
      ...options
    })
  }, [addNotification])

  const showMedicalAlert = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.MEDICAL_ALERT,
      message,
      priority: NOTIFICATION_PRIORITIES.CRITICAL,
      ...options
    })
  }, [addNotification])

  const showDeviceAlert = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.DEVICE_ALERT,
      message,
      priority: NOTIFICATION_PRIORITIES.HIGH,
      ...options
    })
  }, [addNotification])

  const showSystemAlert = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.SYSTEM_ALERT,
      message,
      priority: NOTIFICATION_PRIORITIES.MEDIUM,
      ...options
    })
  }, [addNotification])

  const value = {
    notifications,
    preferences,
    unreadCount,
    addNotification,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAllNotifications,
    updatePreferences,
    loadNotifications,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showMedicalAlert,
    showDeviceAlert,
    showSystemAlert
  }

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  )
}

export default NotificationProvider