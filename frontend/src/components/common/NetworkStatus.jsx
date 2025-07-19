import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Wifi, WifiOff, AlertTriangle } from 'lucide-react'

const NetworkStatus = ({ isOnline }) => {
  const [showStatus, setShowStatus] = useState(false)
  const [wasOffline, setWasOffline] = useState(false)

  useEffect(() => {
    if (!isOnline) {
      setShowStatus(true)
      setWasOffline(true)
    } else if (wasOffline) {
      // Show reconnection message briefly
      setShowStatus(true)
      const timer = setTimeout(() => {
        setShowStatus(false)
      }, 3000)
      
      return () => clearTimeout(timer)
    }
  }, [isOnline, wasOffline])

  const getStatusConfig = () => {
    if (isOnline && wasOffline) {
      return {
        icon: Wifi,
        message: 'Connection restored',
        bgColor: 'bg-green-500',
        textColor: 'text-white',
        iconColor: 'text-white'
      }
    } else if (!isOnline) {
      return {
        icon: WifiOff,
        message: 'No internet connection',
        bgColor: 'bg-red-500',
        textColor: 'text-white',
        iconColor: 'text-white'
      }
    }
    
    return null
  }

  const statusConfig = getStatusConfig()

  if (!showStatus || !statusConfig) {
    return null
  }

  const { icon: Icon, message, bgColor, textColor, iconColor } = statusConfig

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -100 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -100 }}
        className={`fixed top-0 left-0 right-0 z-50 ${bgColor} ${textColor} px-4 py-3 shadow-lg`}
      >
        <div className="flex items-center justify-center gap-2 max-w-7xl mx-auto">
          <Icon className={`w-4 h-4 ${iconColor}`} />
          <span className="text-sm font-medium">{message}</span>
          
          {!isOnline && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.5 }}
              className="ml-2"
            >
              <AlertTriangle className="w-4 h-4 text-yellow-300" />
            </motion.div>
          )}
        </div>
        
        {!isOnline && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="text-center mt-2"
          >
            <p className="text-xs opacity-90">
              Some features may be limited. Data will sync when connection is restored.
            </p>
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  )
}

export default NetworkStatus