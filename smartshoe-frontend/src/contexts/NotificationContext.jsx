import React, { createContext, useContext, useState, useCallback } from 'react'
import { toast } from 'react-hot-toast'

const NotificationContext = createContext()

export const useNotifications = () => {
  const context = useContext(NotificationContext)
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider')
  }
  return context
}

export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info',
  MEDICAL_ALERT: 'medical_alert',
  DEVICE_ALERT: 'device_alert'
}

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([])

  const addNotification = useCallback((notification) => {
    const newNotification = {
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      ...notification
    }

    setNotifications(prev => [newNotification, ...prev.slice(0, 49)]) // Keep last 50
    showToast(newNotification)
    
    return newNotification.id
  }, [])

  const showToast = useCallback((notification) => {
    const { type, title, message, duration = 4000 } = notification
    const text = title || message

    switch (type) {
      case NOTIFICATION_TYPES.SUCCESS:
        toast.success(text, { duration })
        break
      case NOTIFICATION_TYPES.ERROR:
      case NOTIFICATION_TYPES.MEDICAL_ALERT:
        toast.error(text, { duration: duration * 2 })
        break
      case NOTIFICATION_TYPES.WARNING:
      case NOTIFICATION_TYPES.DEVICE_ALERT:
        toast.error(text, { duration: duration * 1.5 })
        break
      case NOTIFICATION_TYPES.INFO:
        toast(text, { duration })
        break
      default:
        toast(text, { duration })
    }
  }, [])

  const showSuccess = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.SUCCESS,
      message,
      ...options
    })
  }, [addNotification])

  const showError = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.ERROR,
      message,
      ...options
    })
  }, [addNotification])

  const showWarning = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.WARNING,
      message,
      ...options
    })
  }, [addNotification])

  const showInfo = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.INFO,
      message,
      ...options
    })
  }, [addNotification])

  const showMedicalAlert = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.MEDICAL_ALERT,
      message,
      ...options
    })
  }, [addNotification])

  const showDeviceAlert = useCallback((message, options = {}) => {
    return addNotification({
      type: NOTIFICATION_TYPES.DEVICE_ALERT,
      message,
      ...options
    })
  }, [addNotification])

  const clearNotifications = useCallback(() => {
    setNotifications([])
  }, [])

  const value = {
    notifications,
    addNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showMedicalAlert,
    showDeviceAlert,
    clearNotifications
  }

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  )
}

export default NotificationProvider