import { useState } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { Lock, Eye, EyeOff, CheckCircle, ArrowLeft } from 'lucide-react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import Button from '@components/common/Button'
import Input from '@components/common/Input'
import { useNotifications } from '@contexts/NotificationContext'

function ResetPassword() {
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)
  
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { showSuccess, showError } = useNotifications()
  
  const token = searchParams.get('token')

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (password !== confirmPassword) {
      showError('Passwords do not match')
      return
    }
    
    if (password.length < 8) {
      showError('Password must be at least 8 characters long')
      return
    }

    setIsLoading(true)
    
    try {
      // API call would go here
      await new Promise(resolve => setTimeout(resolve, 2000)) // Simulate API call
      setIsSuccess(true)
      showSuccess('Password reset successfully!')
      
      setTimeout(() => {
        navigate('/login')
      }, 3000)
    } catch (error) {
      showError('Failed to reset password. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  if (isSuccess) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-md w-full text-center"
        >
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring" }}
              className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mx-auto mb-4"
            >
              <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />
            </motion.div>
            
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Password Reset Successfully!
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              Your password has been updated. You can now log in with your new password.
            </p>
            
            <Button
              onClick={() => navigate('/login')}
              className="w-full"
            >
              Continue to Login
            </Button>
          </div>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center px-4">
      <Helmet>
        <title>Reset Password - Smart Shoe Monitor</title>
        <meta name="description" content="Reset your password for Smart Shoe Monitor" />
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
              <Lock className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </motion.div>
            
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Reset Your Password
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Enter your new password below
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                New Password
              </label>
              <div className="relative">
                <Input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter new password"
                  required
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Confirm Password
              </label>
              <Input
                type={showPassword ? 'text' : 'password'}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm new password"
                required
              />
            </div>

            {/* Password Requirements */}
            <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              <p>Password must contain:</p>
              <ul className="ml-4 space-y-1">
                <li className={password.length >= 8 ? 'text-green-600' : ''}>
                  • At least 8 characters
                </li>
                <li className={/[A-Z]/.test(password) ? 'text-green-600' : ''}>
                  • One uppercase letter
                </li>
                <li className={/[a-z]/.test(password) ? 'text-green-600' : ''}>
                  • One lowercase letter
                </li>
                <li className={/\d/.test(password) ? 'text-green-600' : ''}>
                  • One number
                </li>
              </ul>
            </div>

            <Button
              type="submit"
              disabled={isLoading || !password || !confirmPassword}
              className="w-full"
            >
              {isLoading ? 'Resetting Password...' : 'Reset Password'}
            </Button>
          </form>

          {/* Back to Login */}
          <div className="mt-6 text-center">
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

export default ResetPassword