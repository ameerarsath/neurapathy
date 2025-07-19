import React from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react'

const ErrorFallback = ({ error, resetErrorBoundary }) => {
  const isDevelopment = process.env.NODE_ENV === 'development'

  const handleReload = () => {
    window.location.reload()
  }

  const handleGoHome = () => {
    window.location.href = '/'
  }

  const handleReportBug = () => {
    // In a real app, this would open a bug report form or send to error tracking
    console.error('Error reported:', error)
    const subject = encodeURIComponent('Bug Report: Frontend Error')
    const body = encodeURIComponent(`
Error: ${error.message}
Stack: ${error.stack}
URL: ${window.location.href}
User Agent: ${navigator.userAgent}
Time: ${new Date().toISOString()}
    `)
    
    window.open(`mailto:support@smartshoe.com?subject=${subject}&body=${body}`)
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-md w-full text-center"
      >
        {/* Error icon */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
          className="mx-auto mb-6"
        >
          <div className="w-16 h-16 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center">
            <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
          </div>
        </motion.div>

        {/* Error message */}
        <motion.h1
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-2xl font-bold text-gray-900 dark:text-white mb-2"
        >
          Something went wrong
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="text-gray-600 dark:text-gray-400 mb-6"
        >
          We're sorry for the inconvenience. The application has encountered an unexpected error.
        </motion.p>

        {/* Development error details */}
        {isDevelopment && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ delay: 0.5 }}
            className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6 text-left"
          >
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">
              Development Error Details:
            </h3>
            <p className="text-sm text-red-700 dark:text-red-300 font-mono break-all">
              {error.message}
            </p>
            {error.stack && (
              <details className="mt-2">
                <summary className="text-sm text-red-600 dark:text-red-400 cursor-pointer hover:underline">
                  Stack trace
                </summary>
                <pre className="mt-2 text-xs text-red-600 dark:text-red-400 overflow-x-auto whitespace-pre-wrap">
                  {error.stack}
                </pre>
              </details>
            )}
          </motion.div>
        )}

        {/* Action buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="space-y-3"
        >
          {/* Try again button */}
          <button
            onClick={resetErrorBoundary}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Try Again
          </button>

          {/* Secondary actions */}
          <div className="flex gap-2">
            <button
              onClick={handleReload}
              className="flex-1 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Reload Page
            </button>
            
            <button
              onClick={handleGoHome}
              className="flex-1 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
            >
              <Home className="w-4 h-4" />
              Go Home
            </button>
          </div>

          {/* Report bug button */}
          <button
            onClick={handleReportBug}
            className="w-full bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
          >
            <Bug className="w-4 h-4" />
            Report Bug
          </button>
        </motion.div>

        {/* Help text */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-6 text-sm text-gray-500 dark:text-gray-400"
        >
          <p>
            If this problem persists, please contact our support team at{' '}
            <a
              href="mailto:support@smartshoe.com"
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              support@smartshoe.com
            </a>
          </p>
        </motion.div>

        {/* Error ID for support */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-4 text-xs text-gray-400 dark:text-gray-500 font-mono"
        >
          Error ID: {Date.now().toString(36)}
        </motion.div>
      </motion.div>
    </div>
  )
}

export default ErrorFallback