import React, { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { useAuth } from './AuthContext'
import { useNotifications } from './NotificationContext'

const WebSocketContext = createContext()

export const useWebSocket = () => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [lastMessage, setLastMessage] = useState(null)
  const [hasShownConnectionError, setHasShownConnectionError] = useState(false)
  const [hasShownConnectionSuccess, setHasShownConnectionSuccess] = useState(false)
  
  const { user } = useAuth()
  const { showMedicalAlert, showDeviceAlert, showInfo, showError } = useNotifications()

  const connect = useCallback(() => {
    if (!user || socket) return

    // Skip WebSocket connection if not explicitly enabled
    const enableWebSocket = process.env.NODE_ENV === 'production' || 
                           localStorage.getItem('enableWebSocket') === 'true'
    
    if (!enableWebSocket) {
      console.info('ℹ️ WebSocket disabled in development mode (set localStorage.enableWebSocket=true to enable)')
      setConnectionStatus('disabled')
      return
    }

    try {
      const wsUrl = 'ws://localhost:8080/ws'
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setConnectionStatus('connected')
        setHasShownConnectionError(false)
        if (!hasShownConnectionSuccess) {
          showInfo('Real-time connection established')
          setHasShownConnectionSuccess(true)
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          handleMessage(data)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        setConnectionStatus('disconnected')
        setSocket(null)
        setHasShownConnectionSuccess(false)
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnectionStatus('error')
        if (!hasShownConnectionError) {
          showError('Unable to establish real-time connection. Features may be limited.')
          setHasShownConnectionError(true)
        }
      }

      setSocket(ws)
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      setConnectionStatus('error')
      if (!hasShownConnectionError) {
        showError('Unable to establish real-time connection. Features may be limited.')
        setHasShownConnectionError(true)
      }
    }
  }, [user, socket, showInfo, showError, hasShownConnectionError, hasShownConnectionSuccess])

  const handleMessage = useCallback((data) => {
    setLastMessage(data)
    
    switch (data.type) {
      case 'medical_alert':
        showMedicalAlert(data.message, { title: data.title })
        break
      case 'device_alert':
        showDeviceAlert(data.message, { title: data.title })
        break
      case 'test_result':
        showInfo(`Test result available: ${data.message}`)
        // Dispatch custom event for components to listen to
        window.dispatchEvent(new CustomEvent('testResultReceived', { detail: data }))
        break
      case 'device_data':
        // Dispatch custom event for real-time device data
        window.dispatchEvent(new CustomEvent('deviceDataReceived', { detail: data }))
        break
      default:
        console.log('Unknown message type:', data.type)
    }
  }, [showMedicalAlert, showDeviceAlert, showInfo])

  const sendMessage = useCallback((message) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message))
    }
  }, [socket])

  const disconnect = useCallback(() => {
    if (socket) {
      socket.close()
      setSocket(null)
      setConnectionStatus('disconnected')
    }
  }, [socket])

  useEffect(() => {
    if (user) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [user, connect, disconnect])

  const value = {
    socket,
    connectionStatus,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    isConnected: connectionStatus === 'connected'
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

export default WebSocketProvider