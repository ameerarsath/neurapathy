import { motion } from 'framer-motion'
import { AlertTriangle, RefreshCw, Home, MessageSquare, Phone } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import Button from '@components/common/Button'
import { useState, useEffect } from 'react'

function ServerError() {
  const navigate = useNavigate()
  const [retryCount, setRetryCount] = useState(0)
  const [isRetrying, setIsRetrying] = useState(false)

  const handleRetry = async () => {
    setIsRetrying(true)
    setRetryCount(prev => prev + 1)
    
    // Simulate retry attempt
    setTimeout(() => {
      setIsRetrying(false)
      // In a real app, you would check server status here
      window.location.reload()
    }, 2000)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 to-orange-100 dark:from-gray-900 dark:to-red-900/20 flex items-center justify-center px-4">
      <Helmet>
        <title>Server Error - Smart Shoe Monitor</title>
        <meta name="description" content="We're experiencing technical difficulties. Please try again." />
      </Helmet>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center max-w-lg mx-auto"
      >
        {/* Error Icon Animation */}
        <motion.div
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 150 }}
          className="mb-8"
        >
          <div className="text-8xl font-bold text-red-600 dark:text-red-400 mb-4">
            500
          </div>
          <motion.div
            animate={{ 
              rotate: [0, 5, -5, 0],
              scale: [1, 1.05, 1]
            }}
            transition={{ 
              duration: 2, 
              repeat: Infinity, 
              repeatDelay: 1 
            }}
            className="w-24 h-24 mx-auto mb-6 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center"
          >
            <AlertTriangle className="w-12 h-12 text-red-600 dark:text-red-400" />
          </motion.div>
        </motion.div>

        {/* Content */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Server Error
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-6 leading-relaxed">
            We're experiencing technical difficulties with our medical monitoring system. 
            Our team has been notified and is working to resolve this issue.
          </p>
          
          {retryCount > 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 mb-6"
            >
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                Retry attempt #{retryCount} - Still experiencing issues
              </p>
            </motion.div>
          )}
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="space-y-4"
        >
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button
              onClick={handleRetry}
              disabled={isRetrying}
              className="flex items-center justify-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
              {isRetrying ? 'Retrying...' : 'Try Again'}
            </Button>
            
            <Button
              variant="outline"
              onClick={() => navigate('/')}
              className="flex items-center justify-center gap-2"
            >
              <Home className="w-4 h-4" />
              Go to Dashboard
            </Button>
          </div>
        </motion.div>

        {/* Status Information */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            What can you do?
          </h3>
          
          <div className="grid gap-4 text-left">
            <div className="flex items-start gap-3 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <RefreshCw className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white">
                  Wait and Retry
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Server issues are usually temporary. Try refreshing in a few minutes.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <MessageSquare className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white">
                  Contact Support
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  If the problem persists, our medical support team is standing by.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <Phone className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white">
                  Emergency Access
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  For urgent medical concerns, contact your healthcare provider directly.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Support Actions */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0 }}
          className="mt-8 flex flex-col sm:flex-row gap-3 justify-center"
        >
          <Button
            variant="ghost"
            size="sm"
            className="flex items-center justify-center gap-2"
          >
            <MessageSquare className="w-4 h-4" />
            Contact Support
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            className="flex items-center justify-center gap-2"
          >
            <Phone className="w-4 h-4" />
            Emergency: (555) 123-4567
          </Button>
        </motion.div>

        {/* Error ID for Support */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="mt-8 text-xs text-gray-500 dark:text-gray-400"
        >
          Error ID: {Date.now().toString(36)}-{Math.random().toString(36).substr(2, 9)}
        </motion.div>
      </motion.div>
    </div>
  )
}

export default ServerError