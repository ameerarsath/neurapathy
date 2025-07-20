import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react'
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

    // Temporarily disable WebSocket until backend is deployed with WebSocket support
    const disableWebSocket = true // Force disable for now
    
    if (disableWebSocket) {
      console.log('WebSocket temporarily disabled - backend deployment required')
      setConnectionStatus('disabled')
      return
    }

    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://13.201.120.175:8080/ws'
    
    console.log('🔌 Attempting WebSocket connection to:', wsUrl)
    const newSocket = new WebSocket(wsUrl)

    // Connection event handlers
    newSocket.onopen = () => {
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
    }

    newSocket.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason)
      setConnectionStatus('disconnected')
      
      // Only attempt to reconnect if it was a normal close or network error
      if (event.code !== 1006 && reconnectAttempts < maxReconnectAttempts) {
        console.log(`🔄 Attempting reconnect ${reconnectAttempts + 1}/${maxReconnectAttempts} in ${reconnectDelay}ms`)
        setTimeout(() => {
          setReconnectAttempts(prev => prev + 1)
          const reconnectSocket = initializeSocket()
          setSocket(reconnectSocket)
        }, reconnectDelay * (reconnectAttempts + 1))
      } else if (event.code === 1006) {
        console.log('❌ WebSocket connection failed (code 1006) - not attempting reconnect')
        if (!hasShownConnectionError) {
          showSystemAlert('WebSocket connection failed. Real-time features disabled.')
          setHasShownConnectionError(true)
        }
      }
    }

    newSocket.onerror = (error) => {
      console.error('WebSocket connection error:', error)
      setConnectionStatus('error')
      setReconnectAttempts(prev => prev + 1)
      
      if (!hasShownConnectionError) {
        showSystemAlert('Unable to establish real-time connection. Features may be limited.')
        setHasShownConnectionError(true)
      }
    }

    // Message handler for all incoming WebSocket messages
    newSocket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        console.log('WebSocket message received:', message)
        
        // Handle different message types
        switch (message.type) {
          case 'welcome':
            console.log('Welcome message received:', message)
            break
          case 'pong':
            handlePongMessage(message)
            break
          case 'medical_alert':
            handleMedicalAlert(message.data || message)
            break
          case 'device_alert':
            handleDeviceAlert(message.data || message)
            break
          case 'system_alert':
            handleSystemAlert(message.data || message)
            break
          case 'device_data':
            handleDeviceData(message.data || message)
            break
          case 'test_result':
            handleTestResult(message.data || message)
            break
          case 'ml_prediction':
            handleMLPrediction(message.data || message)
            break
          default:
            console.log('Unknown message type:', message.type)
        }
        
        // Update last message and history
        setLastMessage({ type: message.type, data: message, timestamp: Date.now() })
        setMessageHistory(prev => [...prev.slice(-99), { type: message.type, data: message, timestamp: Date.now() }])
        
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    // Handle pong messages for latency measurement  
    const handlePongMessage = (message) => {
      if (message.originalTimestamp) {
        const now = Date.now()
        const originalTime = new Date(message.originalTimestamp).getTime()
        setLatency(now - originalTime)
      }
    }


    setSocket(newSocket)
    return newSocket
  }, [user, token, showMedicalAlert, showDeviceAlert, showSystemAlert, showInfo, hasShownConnectionError, hasShownConnectionSuccess])

  // Start ping/pong for latency measurement
  const startPingPong = useCallback((socket) => {
    if (pingTimer.current) {
      clearInterval(pingTimer.current)
    }

    pingTimer.current = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
          type: 'ping',
          timestamp: new Date().toISOString()
        }))
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
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
          type: 'subscribe',
          channel: channel
        }))
        setSubscribedChannels(prev => new Set([...prev, channel]))
      }
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
        if (newSocket && newSocket.readyState === WebSocket.OPEN) {
          newSocket.close()
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
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close()
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
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: 'subscribe',
        channel: channel
      }))
      setSubscribedChannels(prev => new Set([...prev, channel]))
    }
  }, [socket])

  const unsubscribe = useCallback((channel) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: 'unsubscribe',
        channel: channel
      }))
      setSubscribedChannels(prev => {
        const newSet = new Set(prev)
        newSet.delete(channel)
        return newSet
      })
    }
  }, [socket])

  const sendMessage = useCallback((type, data) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: type,
        data: data,
        timestamp: new Date().toISOString()
      }))
    }
  }, [socket])

  const getConnectionStatus = useCallback(() => {
    return {
      status: connectionStatus,
      connected: socket?.readyState === WebSocket.OPEN || false,
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