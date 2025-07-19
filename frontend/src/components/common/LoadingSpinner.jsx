import React from 'react'
import { motion } from 'framer-motion'

const LoadingSpinner = ({ 
  size = 'md', 
  color = 'primary', 
  text = 'Loading...', 
  fullscreen = false,
  className = ''
}) => {
  // Size variants
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  }

  // Color variants
  const colorClasses = {
    primary: 'border-blue-500',
    secondary: 'border-gray-500',
    success: 'border-green-500',
    warning: 'border-yellow-500',
    error: 'border-red-500',
    white: 'border-white'
  }

  // Animation variants
  const spinnerVariants = {
    animate: {
      rotate: 360,
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: 'linear'
      }
    }
  }

  const pulseVariants = {
    animate: {
      scale: [1, 1.2, 1],
      opacity: [1, 0.8, 1],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    }
  }

  const SpinnerContent = () => (
    <div className={`flex flex-col items-center justify-center ${className}`}>
      {/* Main spinner */}
      <motion.div
        className={`
          ${sizeClasses[size]}
          border-2 border-t-transparent border-r-transparent
          ${colorClasses[color]}
          rounded-full
        `}
        variants={spinnerVariants}
        animate="animate"
      />
      
      {/* Pulse effect */}
      <motion.div
        className={`
          ${sizeClasses[size]}
          border-2 border-transparent
          ${colorClasses[color]}
          rounded-full
          absolute
          opacity-20
        `}
        variants={pulseVariants}
        animate="animate"
      />
      
      {/* Loading text */}
      {text && (
        <motion.p
          className="mt-4 text-sm text-gray-600 dark:text-gray-400"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {text}
        </motion.p>
      )}
    </div>
  )

  // Fullscreen loader
  if (fullscreen) {
    return (
      <motion.div
        className="fixed inset-0 bg-white dark:bg-gray-900 bg-opacity-90 flex items-center justify-center z-50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <div className="text-center">
          <SpinnerContent />
          <motion.div
            className="mt-8 w-32 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <motion.div
              className="h-full bg-blue-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: '100%' }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut'
              }}
            />
          </motion.div>
        </div>
      </motion.div>
    )
  }

  return (
    <div className="relative inline-flex items-center justify-center">
      <SpinnerContent />
    </div>
  )
}

// Specialized loading spinners
export const ButtonSpinner = ({ size = 'sm', color = 'white' }) => (
  <LoadingSpinner size={size} color={color} text="" />
)

export const PageSpinner = ({ text = 'Loading page...' }) => (
  <LoadingSpinner size="lg" text={text} fullscreen />
)

export const InlineSpinner = ({ size = 'sm', color = 'primary' }) => (
  <LoadingSpinner size={size} color={color} text="" className="inline-block" />
)

export const CardSpinner = ({ text = 'Loading...' }) => (
  <div className="flex items-center justify-center p-8">
    <LoadingSpinner size="md" text={text} />
  </div>
)

export default LoadingSpinner