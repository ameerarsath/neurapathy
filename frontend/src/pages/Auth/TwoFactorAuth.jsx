import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { Shield, RefreshCw, ArrowLeft } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import Button from '@components/common/Button'
import { useNotifications } from '@contexts/NotificationContext'

function TwoFactorAuth() {
  const [code, setCode] = useState(['', '', '', '', '', ''])
  const [isLoading, setIsLoading] = useState(false)
  const [timeLeft, setTimeLeft] = useState(30)
  const [canResend, setCanResend] = useState(false)
  
  const navigate = useNavigate()
  const { showSuccess, showError } = useNotifications()
  const inputRefs = useRef([])

  // Countdown timer
  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
      return () => clearTimeout(timer)
    } else {
      setCanResend(true)
    }
  }, [timeLeft])

  const handleCodeChange = (index, value) => {
    if (value.length > 1) return
    
    const newCode = [...code]
    newCode[index] = value
    setCode(newCode)
    
    // Auto-focus next input
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus()
    }
    
    // Auto-submit when all fields are filled
    if (newCode.every(digit => digit !== '') && newCode.join('').length === 6) {
      handleSubmit(newCode.join(''))
    }
  }

  const handleKeyDown = (index, e) => {
    if (e.key === 'Backspace' && !code[index] && index > 0) {
      inputRefs.current[index - 1]?.focus()
    }
  }

  const handleSubmit = async (verificationCode = code.join('')) => {
    if (verificationCode.length !== 6) {
      showError('Please enter the complete 6-digit code')
      return
    }

    setIsLoading(true)
    
    try {
      // API call would go here
      await new Promise(resolve => setTimeout(resolve, 2000)) // Simulate API call
      
      if (verificationCode === '123456') { // Demo code
        showSuccess('Authentication successful!')
        navigate('/dashboard')
      } else {
        showError('Invalid verification code. Please try again.')
        setCode(['', '', '', '', '', ''])
        inputRefs.current[0]?.focus()
      }
    } catch (error) {
      showError('Verification failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleResendCode = async () => {
    if (!canResend) return
    
    try {
      // API call would go here
      await new Promise(resolve => setTimeout(resolve, 1000))
      showSuccess('Verification code sent!')
      setTimeLeft(30)
      setCanResend(false)
      setCode(['', '', '', '', '', ''])
      inputRefs.current[0]?.focus()
    } catch (error) {
      showError('Failed to resend code. Please try again.')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center px-4">
      <Helmet>
        <title>Two-Factor Authentication - Smart Shoe Monitor</title>
        <meta name="description" content="Complete two-factor authentication" />
      </Helmet>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-md w-full"
      >
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.1 }}
              className="w-12 h-12 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center mx-auto mb-4"
            >
              <Shield className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </motion.div>
            
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Two-Factor Authentication
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              We've sent a 6-digit verification code to your email
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              user@example.com
            </p>
          </div>

          {/* Code Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4 text-center">
              Enter Verification Code
            </label>
            
            <div className="flex justify-center gap-2 mb-4">
              {code.map((digit, index) => (
                <motion.input
                  key={index}
                  ref={el => inputRefs.current[index] = el}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: index * 0.1 }}
                  type="text"
                  inputMode="numeric"
                  maxLength={1}
                  value={digit}
                  onChange={(e) => handleCodeChange(index, e.target.value.replace(/\D/g, ''))}
                  onKeyDown={(e) => handleKeyDown(index, e)}
                  className="w-12 h-12 text-center text-lg font-semibold border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  disabled={isLoading}
                />
              ))}
            </div>

            {/* Timer */}
            <div className="text-center text-sm text-gray-600 dark:text-gray-400">
              {canResend ? (
                <button
                  onClick={handleResendCode}
                  className="text-blue-600 hover:text-blue-500 font-medium flex items-center justify-center gap-1 mx-auto"
                >
                  <RefreshCw className="w-4 h-4" />
                  Resend Code
                </button>
              ) : (
                <span>Resend code in {timeLeft}s</span>
              )}
            </div>
          </div>

          {/* Submit Button */}
          <Button
            onClick={() => handleSubmit()}
            disabled={isLoading || code.some(digit => digit === '')}
            className="w-full mb-4"
          >
            {isLoading ? 'Verifying...' : 'Verify Code'}
          </Button>

          {/* Help Text */}
          <div className="text-center text-sm text-gray-600 dark:text-gray-400 mb-4">
            <p>Didn't receive the code?</p>
            <ul className="mt-2 space-y-1">
              <li>• Check your spam/junk folder</li>
              <li>• Make sure your email address is correct</li>
              <li>• Try requesting a new code</li>
            </ul>
          </div>

          {/* Demo Code Hint */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 mb-4">
            <p className="text-xs text-blue-800 dark:text-blue-200 text-center">
              <strong>Demo:</strong> Use code <code>123456</code> to continue
            </p>
          </div>

          {/* Back to Login */}
          <div className="text-center">
            <button
              onClick={() => navigate('/login')}
              className="text-sm text-blue-600 hover:text-blue-500 flex items-center justify-center gap-1 mx-auto"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Login
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default TwoFactorAuth