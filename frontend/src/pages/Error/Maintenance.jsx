import { motion } from 'framer-motion'
import { Settings, Clock, CheckCircle, AlertTriangle, Phone } from 'lucide-react'
import { Helmet } from 'react-helmet-async'
import { useState, useEffect } from 'react'

function Maintenance() {
  const [timeRemaining, setTimeRemaining] = useState({
    hours: 2,
    minutes: 30,
    seconds: 0
  })

  // Countdown timer
  useEffect(() => {
    const timer = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev.seconds > 0) {
          return { ...prev, seconds: prev.seconds - 1 }
        } else if (prev.minutes > 0) {
          return { ...prev, minutes: prev.minutes - 1, seconds: 59 }
        } else if (prev.hours > 0) {
          return { hours: prev.hours - 1, minutes: 59, seconds: 59 }
        }
        return prev
      })
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-yellow-100 dark:from-gray-900 dark:to-orange-900/20 flex items-center justify-center px-4">
      <Helmet>
        <title>System Maintenance - Smart Shoe Monitor</title>
        <meta name="description" content="We're performing scheduled maintenance to improve your experience" />
      </Helmet>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center max-w-2xl mx-auto"
      >
        {/* Maintenance Icon Animation */}
        <motion.div
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 150 }}
          className="mb-8"
        >
          <motion.div
            animate={{ 
              rotate: [0, 360],
            }}
            transition={{ 
              duration: 8, 
              repeat: Infinity,
              ease: "linear"
            }}
            className="w-32 h-32 mx-auto mb-6 bg-orange-100 dark:bg-orange-900/20 rounded-full flex items-center justify-center"
          >
            <Settings className="w-16 h-16 text-orange-600 dark:text-orange-400" />
          </motion.div>
        </motion.div>

        {/* Content */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Scheduled Maintenance
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
            We're upgrading our medical monitoring system to serve you better. 
            Thank you for your patience.
          </p>
        </motion.div>

        {/* Countdown Timer */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.6 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 mb-8"
        >
          <div className="flex items-center justify-center gap-2 mb-4">
            <Clock className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Estimated Time Remaining
            </h3>
          </div>
          
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                {timeRemaining.hours.toString().padStart(2, '0')}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Hours</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                {timeRemaining.minutes.toString().padStart(2, '0')}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Minutes</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                {timeRemaining.seconds.toString().padStart(2, '0')}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Seconds</div>
            </div>
          </div>
        </motion.div>

        {/* Maintenance Details */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="grid md:grid-cols-2 gap-6 mb-8"
        >
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              What We're Improving
            </h3>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• Enhanced ML prediction algorithms</li>
              <li>• Faster test result processing</li>
              <li>• Improved device connectivity</li>
              <li>• Enhanced security measures</li>
              <li>• Performance optimizations</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
              Important Notes
            </h3>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• All patient data remains secure</li>
              <li>• Emergency services are available</li>
              <li>• Device pairing will resume automatically</li>
              <li>• Recent test data is safely stored</li>
              <li>• No action required from users</li>
            </ul>
          </div>
        </motion.div>

        {/* Emergency Contact */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0 }}
          className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6"
        >
          <div className="flex items-center justify-center gap-2 mb-3">
            <Phone className="w-5 h-5 text-red-600 dark:text-red-400" />
            <h3 className="text-lg font-semibold text-red-800 dark:text-red-200">
              Emergency Medical Support
            </h3>
          </div>
          <p className="text-red-700 dark:text-red-300 mb-4">
            For urgent medical concerns during maintenance, contact our 24/7 emergency line:
          </p>
          <div className="text-2xl font-bold text-red-800 dark:text-red-200">
            (555) 911-HELP
          </div>
          <p className="text-sm text-red-600 dark:text-red-400 mt-2">
            Available for patients and caregivers
          </p>
        </motion.div>

        {/* Progress Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="mt-8"
        >
          <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>Maintenance Progress</span>
            <span>75% Complete</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: '75%' }}
              transition={{ duration: 2, ease: "easeOut" }}
              className="bg-orange-600 h-2 rounded-full"
            />
          </div>
        </motion.div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.4 }}
          className="mt-8 text-sm text-gray-500 dark:text-gray-400"
        >
          <p>Thank you for using Smart Shoe Health Monitor</p>
          <p>We'll be back online shortly with exciting improvements!</p>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default Maintenance