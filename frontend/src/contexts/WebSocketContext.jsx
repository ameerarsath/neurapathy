import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react'
import { io } from 'socket.io-client'
import { useNotifications } from './NotificationContext'
import { useAuth } from './AuthContext'

// Create context
const WebSocketContext = createContext()

// Custom hook to use WebSocket
export const useWebSocket = () => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

// WebSocket event types
export const WS_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  DEVICE_DATA: 'device_data',
  MEDICAL_ALERT: 'medical_alert',
  DEVICE_ALERT: 'device_alert',
  SYSTEM_ALERT: 'system_alert',
  TEST_RESULT: 'test_result',
  PATIENT_UPDATE: 'patient_update',
  ML_PREDICTION: 'ml_prediction',
  BATTERY_LOW: 'battery_low',
  DEVICE_OFFLINE: 'device_offline',
  ANOMALY_DETECTED: 'anomaly_detected',
  THRESHOLD_EXCEEDED: 'threshold_exceeded',
  CALIBRATION_REQUIRED: 'calibration_required',
  SYSTEM_MAINTENANCE: 'system_maintenance',
  USER_SESSION_UPDATE: 'user_session_update'
}

// WebSocket provider component
export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [lastMessage, setLastMessage] = useState(null)
  const [subscribedChannels, setSubscribedChannels] = useState(new Set())
  const [messageHistory, setMessageHistory] = useState([])
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const [latency, setLatency] = useState(0)
  const [hasShownConnectionError, setHasShownConnectionError] = useState(false)
  const [hasShownConnectionSuccess, setHasShownConnectionSuccess] = useState(false)
  
  const { user, token } = useAuth()
  const { showMedicalAlert, showDeviceAlert, showSystemAlert, showInfo } = useNotifications()
  
  const reconnectTimer = useRef(null)
  const pingTimer = useRef(null)
  const maxReconnectAttempts = 5
  const reconnectDelay = 1000

  // Initialize WebSocket connection
  const initializeSocket = useCallback(() => {
    if (!user || !token) return

    // Skip WebSocket connection if not explicitly enabled
    const enableWebSocket = import.meta.env.VITE_APP_ENVIRONMENT === 'production' || 
                           localStorage.getItem('enableWebSocket') === 'true'
    
    if (!enableWebSocket) {
      console.log('WebSocket disabled in development mode')
      setConnectionStatus('disabled')
      return
    }

    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8080'
    
    const newSocket = io(wsUrl, {
      auth: {
        token: token
      },
      transports: ['websocket'],
      upgrade: false,
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: maxReconnectAttempts,
      reconnectionDelay: reconnectDelay,
      timeout: 10000,
      forceNew: true
    })

    // Connection event handlers
    newSocket.on('connect', () => {
      console.log('WebSocket connected')
      setConnectionStatus('connected')
      setReconnectAttempts(0)
      setHasShownConnectionError(false)
      
      // Start ping/pong for latency measurement
      startPingPong(newSocket)
      
      // Subscribe to user-specific channels
      subscribeToUserChannels(newSocket)
      
      if (!hasShownConnectionSuccess) {
        showInfo('Real-time connection established')
        setHasShownConnectionSuccess(true)
      }
    })

    newSocket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason)
      setConnectionStatus('disconnected')
      
      if (reason === 'io server disconnect') {
        // Server disconnected the client, reconnect manually
        setTimeout(() => {
          newSocket.connect()
        }, reconnectDelay)
      }
    })

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error)
      setConnectionStatus('error')
      setReconnectAttempts(prev => prev + 1)
      
      if (!hasShownConnectionError) {
        showSystemAlert('Unable to establish real-time connection. Features may be limited.')
        setHasShownConnectionError(true)
      }
    })

    newSocket.on('reconnect', (attemptNumber) => {
      console.log(`WebSocket reconnected after ${attemptNumber} attempts`)
      setConnectionStatus('connected')
      setReconnectAttempts(0)
      setHasShownConnectionError(false)
      showInfo('Connection restored')
    })

    newSocket.on('reconnect_attempt', (attemptNumber) => {
      console.log(`WebSocket reconnection attempt ${attemptNumber}`)
      setConnectionStatus('reconnecting')
    })

    newSocket.on('reconnect_error', (error) => {
      console.error('WebSocket reconnection error:', error)
      setConnectionStatus('error')
    })

    newSocket.on('reconnect_failed', () => {
      console.error('WebSocket reconnection failed')
      setConnectionStatus('failed')
      if (!hasShownConnectionError) {
        showSystemAlert('Connection failed. Please refresh the page.')
        setHasShownConnectionError(true)
      }
    })

    // Pong handler for latency measurement
    newSocket.on('pong', (timestamp) => {
      const now = Date.now()
      setLatency(now - timestamp)
    })

    // Medical alert handler
    newSocket.on(WS_EVENTS.MEDICAL_ALERT, (data) => {
      console.log('Medical alert received:', data)
      handleMedicalAlert(data)
    })

    // Device alert handler
    newSocket.on(WS_EVENTS.DEVICE_ALERT, (data) => {
      console.log('Device alert received:', data)
      handleDeviceAlert(data)
    })

    // System alert handler
    newSocket.on(WS_EVENTS.SYSTEM_ALERT, (data) => {
      console.log('System alert received:', data)
      handleSystemAlert(data)
    })

    // Device data handler
    newSocket.on(WS_EVENTS.DEVICE_DATA, (data) => {
      console.log('Device data received:', data)
      handleDeviceData(data)
    })

    // Test result handler
    newSocket.on(WS_EVENTS.TEST_RESULT, (data) => {
      console.log('Test result received:', data)
      handleTestResult(data)
    })

    // ML prediction handler
    newSocket.on(WS_EVENTS.ML_PREDICTION, (data) => {
      console.log('ML prediction received:', data)
      handleMLPrediction(data)
    })

    // Battery low handler
    newSocket.on(WS_EVENTS.BATTERY_LOW, (data) => {
      console.log('Battery low alert:', data)
      handleBatteryLow(data)
    })

    // Device offline handler
    newSocket.on(WS_EVENTS.DEVICE_OFFLINE, (data) => {
      console.log('Device offline:', data)
      handleDeviceOffline(data)
    })

    // Anomaly detected handler
    newSocket.on(WS_EVENTS.ANOMALY_DETECTED, (data) => {
      console.log('Anomaly detected:', data)
      handleAnomalyDetected(data)
    })

    // Threshold exceeded handler
    newSocket.on(WS_EVENTS.THRESHOLD_EXCEEDED, (data) => {
      console.log('Threshold exceeded:', data)
      handleThresholdExceeded(data)
    })

    // Calibration required handler
    newSocket.on(WS_EVENTS.CALIBRATION_REQUIRED, (data) => {
      console.log('Calibration required:', data)
      handleCalibrationRequired(data)
    })

    // System maintenance handler
    newSocket.on(WS_EVENTS.SYSTEM_MAINTENANCE, (data) => {
      console.log('System maintenance:', data)
      handleSystemMaintenance(data)
    })

    // User session update handler
    newSocket.on(WS_EVENTS.USER_SESSION_UPDATE, (data) => {
      console.log('User session update:', data)
      handleUserSessionUpdate(data)
    })

    setSocket(newSocket)
    return newSocket
  }, [user, token, showMedicalAlert, showDeviceAlert, showSystemAlert, showInfo, hasShownConnectionError, hasShownConnectionSuccess])

  // Start ping/pong for latency measurement
  const startPingPong = useCallback((socket) => {
    if (pingTimer.current) {
      clearInterval(pingTimer.current)
    }

    pingTimer.current = setInterval(() => {
      if (socket.connected) {
        socket.emit('ping', Date.now())
      }
    }, 30000) // Ping every 30 seconds
  }, [])

  // Subscribe to user-specific channels
  const subscribeToUserChannels = useCallback((socket) => {
    if (!user) return

    const userChannels = [
      `user:${user.id}`,
      `role:${user.role}`,
      'system:alerts',
      'medical:alerts'
    ]

    // Subscribe to patient-specific channels if user is a patient
    if (user.role === 'PATIENT') {
      userChannels.push(`patient:${user.id}`)
    }

    // Subscribe to provider-specific channels if user is a healthcare provider
    if (user.role === 'HEALTHCARE_PROVIDER') {
      userChannels.push(`provider:${user.id}`)
    }

    userChannels.forEach(channel => {
      socket.emit('subscribe', channel)
      setSubscribedChannels(prev => new Set([...prev, channel]))
    })
  }, [user])

  // Event handlers
  const handleMedicalAlert = useCallback((data) => {
    setLastMessage({ type: WS_EVENTS.MEDICAL_ALERT, data, timestamp: Date.now() })
    setMessageHistory(prev => [...prev.slice(-99), { type: WS_EVENTS.MEDICAL_ALERT, data, timestamp: Date.now() }])
    
    showMedicalAlert(data.message, {
      title: data.title,
      priority: data.priority,
      data: data
    })
  }, [showMedicalAlert])

  const handleDeviceAlert = useCallback((data) => {
    setLastMessage({ type: WS_EVENTS.DEVICE_ALERT, data, timestamp: Date.now() })
    setMessageHistory(prev => [...prev.slice(-99), { type: WS_EVENTS.DEVICE_ALERT, data, timestamp: Date.now() }])
    
    showDeviceAlert(data.message, {
      title: data.title,
      priority: data.priority,
      data: data
    })
  }, [showDeviceAlert])

  const handleSystemAlert = useCallback((data) => {
    setLastMessage({ type: WS_EVENTS.SYSTEM_ALERT, data, timestamp: Date.now() })
    setMessageHistory(prev => [...prev.slice(-99), { type: WS_EVENTS.SYSTEM_ALERT, data, timestamp: Date.now() }])
    
    showSystemAlert(data.message, {
      title: data.title,
      priority: data.priority,
      data: data
    })
  }, [showSystemAlert])

  const handleDeviceData = useCallback((data) => {
    setLastMessage({ type: WS_EVENTS.DEVICE_DATA, data, timestamp: Date.now() })
    setMessageHistory(prev => [...prev.slice(-99), { type: WS_EVENTS.DEVICE_DATA, data, timestamp: Date.now() }])
    
    // Emit custom event for device data
    window.dispatchEvent(new CustomEvent('deviceDataReceived', { detail: data }))
  }, [])

  const handleTestResult = useCallback((data) => {
    setLastMessage({ type: WS_EVENTS.TEST_RESULT, data, timestamp: Date.now() })
    setMessageHistory(prev => [...prev.slice(-99), { type: WS_EVENTS.TEST_RESULT, data, timestamp: Date.now() }])
    
    showInfo(`Test result available for ${data.patientName}`, {
      title: 'Test Completed',
      data: data
    })
  }, [showInfo])

  const handleMLPrediction = useCallback((data) => {
    setLastMessage({ type: WS_EVENTS.ML_PREDICTION, data, timestamp: Date.now() })
    setMessageHistory(prev => [...prev.slice(-99), { type: WS_EVENTS.ML_PREDICTION, data, timestamp: Date.now() }])
    
    // Emit custom event for ML prediction
    window.dispatchEvent(new CustomEvent('mlPredictionReceived', { detail: data }))
  }, [])

  const handleBatteryLow = useCallback((data) => {
    showDeviceAlert(`Device ${data.deviceId} battery is low (${data.batteryLevel}%)`, {
      title: 'Battery Low',
      priority: 'HIGH',
      data: data
    })
  }, [showDeviceAlert])

  const handleDeviceOffline = useCallback((data) => {
    showDeviceAlert(`Device ${data.deviceId} is offline`, {
      title: 'Device Offline',
      priority: 'HIGH',
      data: data
    })
  }, [showDeviceAlert])

  const handleAnomalyDetected = useCallback((data) => {
    showMedicalAlert(`Anomaly detected: ${data.description}`, {
      title: 'Anomaly Alert',
      priority: 'CRITICAL',
      data: data
    })
  }, [showMedicalAlert])

  const handleThresholdExceeded = useCallback((data) => {
    showMedicalAlert(`Threshold exceeded: ${data.parameter} = ${data.value}`, {
      title: 'Threshold Alert',
      priority: 'HIGH',
      data: data
    })
  }, [showMedicalAlert])

  const handleCalibrationRequired = useCallback((data) => {
    showDeviceAlert(`Device ${data.deviceId} requires calibration`, {
      title: 'Calibration Required',
      priority: 'MEDIUM',
      data: data
    })
  }, [showDeviceAlert])

  const handleSystemMaintenance = useCallback((data) => {
    showSystemAlert(data.message, {
      title: 'System Maintenance',
      priority: 'MEDIUM',
      data: data
    })
  }, [showSystemAlert])

  const handleUserSessionUpdate = useCallback((data) => {
    // Handle user session updates (e.g., permissions changed)
    window.dispatchEvent(new CustomEvent('userSessionUpdate', { detail: data }))
  }, [])

  // Initialize socket when user changes
  useEffect(() => {
    if (user && token) {
      const newSocket = initializeSocket()
      
      return () => {
        if (newSocket) {
          newSocket.disconnect()
        }
        if (pingTimer.current) {
          clearInterval(pingTimer.current)
        }
        if (reconnectTimer.current) {
          clearTimeout(reconnectTimer.current)
        }
      }
    }
  }, [user, token, initializeSocket])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (socket) {
        socket.disconnect()
      }
      if (pingTimer.current) {
        clearInterval(pingTimer.current)
      }
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
      }
    }
  }, [socket])

  // Public methods
  const subscribe = useCallback((channel) => {
    if (socket && socket.connected) {
      socket.emit('subscribe', channel)
      setSubscribedChannels(prev => new Set([...prev, channel]))
    }
  }, [socket])

  const unsubscribe = useCallback((channel) => {
    if (socket && socket.connected) {
      socket.emit('unsubscribe', channel)
      setSubscribedChannels(prev => {
        const newSet = new Set(prev)
        newSet.delete(channel)
        return newSet
      })
    }
  }, [socket])

  const sendMessage = useCallback((event, data) => {
    if (socket && socket.connected) {
      socket.emit(event, data)
    }
  }, [socket])

  const getConnectionStatus = useCallback(() => {
    return {
      status: connectionStatus,
      connected: socket?.connected || false,
      latency,
      reconnectAttempts
    }
  }, [connectionStatus, socket, latency, reconnectAttempts])

  const value = {
    socket,
    connectionStatus,
    lastMessage,
    messageHistory,
    subscribedChannels: Array.from(subscribedChannels),
    latency,
    reconnectAttempts,
    subscribe,
    unsubscribe,
    sendMessage,
    getConnectionStatus
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

export default WebSocketProvider