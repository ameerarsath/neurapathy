import { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { Mail, ArrowLeft, AlertCircle, CheckCircle } from 'lucide-react'

import { useAuth } from '@contexts/AuthContext'
import { useNotifications } from '@contexts/NotificationContext'
import Button from '@components/common/Button'
import Input from '@components/common/Input'

function ForgotPassword() {
  const { forgotPassword } = useAuth()
  const { showError } = useNotifications()
  
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [email, setEmail] = useState('')
  const [errors, setErrors] = useState({})

  const validateForm = () => {
    const newErrors = {}
    
    if (!email) {
      newErrors.email = 'Email is required'
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = 'Please enter a valid email address'
    }
    
    return newErrors
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    const formErrors = validateForm()
    if (Object.keys(formErrors).length > 0) {
      setErrors(formErrors)
      return
    }
    
    setLoading(true)
    setErrors({})

    try {
      await forgotPassword(email.toLowerCase().trim())
      setSuccess(true)
    } catch (error) {
      console.error('Forgot password error:', error)
      const errorMessage = error.response?.data?.message || 'Failed to send reset email. Please try again.'
      
      if (error.response?.status === 404) {
        setErrors({ 
          email: 'No account found with this email address.' 
        })
      } else if (error.response?.status === 429) {
        setErrors({ 
          general: 'Too many requests. Please wait before trying again.' 
        })
      } else {
        setErrors({ 
          general: errorMessage 
        })
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center px-4">
      <Helmet>
        <title>Forgot Password - Smart Shoe Monitor</title>
        <meta name="description" content="Reset your Smart Shoe monitoring account password" />
      </Helmet>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md"
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
              <Mail className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </motion.div>
            
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Reset Password
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Enter your email address and we'll send you instructions to reset your password
            </p>
          </div>

          {success ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center"
            >
              <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6 mb-6">
                <CheckCircle className="w-12 h-12 text-green-600 dark:text-green-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-green-800 dark:text-green-200 mb-2">
                  Check Your Email
                </h3>
                <p className="text-sm text-green-700 dark:text-green-300">
                  Password reset instructions have been sent to <strong>{email}</strong>
                </p>
              </div>
              
              <div className="space-y-4">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Didn't receive the email? Check your spam folder or try again in a few minutes.
                </p>
                
                <div className="flex flex-col gap-3">
                  <Button
                    onClick={() => setSuccess(false)}
                    variant="outline"
                    className="w-full"
                  >
                    Try Different Email
                  </Button>
                  
                  <Link 
                    to="/login" 
                    className="inline-flex items-center justify-center gap-2 text-blue-600 hover:text-blue-500 dark:text-blue-400 font-medium"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    Return to Login
                  </Link>
                </div>
              </div>
            </motion.div>
          ) : (
            <>
              {/* Form */}
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* General Error */}
                {errors.general && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4"
                  >
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                      <span className="text-sm text-red-800 dark:text-red-200">
                        {errors.general}
                      </span>
                    </div>
                  </motion.div>
                )}

                {/* Email Field */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Email Address
                  </label>
                  <div className="relative">
                    <Input
                      type="email"
                      name="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="Enter your email address"
                      required
                      className={`pl-10 ${errors.email ? 'border-red-500' : ''}`}
                      disabled={loading}
                    />
                    <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  </div>
                  {errors.email && (
                    <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                      {errors.email}
                    </p>
                  )}
                </div>

                {/* Submit Button */}
                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full"
                >
                  {loading ? 'Sending Instructions...' : 'Send Reset Instructions'}
                </Button>
              </form>

              {/* Back to Login */}
              <div className="mt-6 text-center">
                <Link 
                  to="/login" 
                  className="inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-500 dark:text-blue-400 font-medium"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to Login
                </Link>
              </div>

              {/* Support Link */}
              <div className="mt-4 text-center">
                <Link 
                  to="/support" 
                  className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  Need help? Contact Support
                </Link>
              </div>
            </>
          )}
        </div>
      </motion.div>
    </div>
  )
}

export default ForgotPassword