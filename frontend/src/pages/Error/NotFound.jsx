import { motion } from 'framer-motion'
import { Home, ArrowLeft, Search, HelpCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import Button from '@components/common/Button'

function NotFound() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center px-4">
      <Helmet>
        <title>Page Not Found - Smart Shoe Monitor</title>
        <meta name="description" content="The page you're looking for doesn't exist" />
      </Helmet>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center max-w-md mx-auto"
      >
        {/* 404 Animation */}
        <motion.div
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 150 }}
          className="mb-8"
        >
          <div className="text-8xl font-bold text-blue-600 dark:text-blue-400 mb-4">
            404
          </div>
          <motion.div
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
            className="w-24 h-24 mx-auto mb-6 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center"
          >
            <Search className="w-12 h-12 text-blue-600 dark:text-blue-400" />
          </motion.div>
        </motion.div>

        {/* Content */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Page Not Found
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
            The page you're looking for doesn't exist or has been moved. 
            Let's get you back to monitoring your health.
          </p>
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
              onClick={() => navigate('/')}
              className="flex items-center justify-center gap-2"
            >
              <Home className="w-4 h-4" />
              Go to Dashboard
            </Button>
            
            <Button
              variant="outline"
              onClick={() => navigate(-1)}
              className="flex items-center justify-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Go Back
            </Button>
          </div>

          <Button
            variant="ghost"
            onClick={() => navigate('/help')}
            className="flex items-center justify-center gap-2 text-sm"
          >
            <HelpCircle className="w-4 h-4" />
            Need Help?
          </Button>
        </motion.div>

        {/* Helpful Links */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700"
        >
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
            You might be looking for:
          </h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <button
              onClick={() => navigate('/test-sessions')}
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              Start New Test
            </button>
            <button
              onClick={() => navigate('/test-results')}
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              View Results
            </button>
            <button
              onClick={() => navigate('/devices')}
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              Device Settings
            </button>
            <button
              onClick={() => navigate('/settings')}
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              Account Settings
            </button>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default NotFound