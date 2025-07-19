import React, { useState } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import LoadingSpinner from '../../components/common/LoadingSpinner'
import { Eye, EyeOff, Stethoscope, Shield, User, Zap, Key, Smartphone } from 'lucide-react'
import { smartShoeAPI } from '../../services/api'
import api from '../../services/api'
import toast from 'react-hot-toast'

const Login = () => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
    totpCode: '',
    backupCode: ''
  })
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [requiresTwoFactor, setRequiresTwoFactor] = useState(false)
  const [useBackupCode, setUseBackupCode] = useState(false)
  
  const { login } = useAuth()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    
    try {
      // First attempt login with username/password
      const response = await smartShoeAPI.auth.login({
        username: credentials.username,
        password: credentials.password,
        totpCode: credentials.totpCode,
        backupCode: credentials.backupCode
      })

      if (response.data.success) {
        // Login successful - set user state directly from the API response
        const token = btoa(`${credentials.username}:${credentials.password}`)
        localStorage.setItem('smartshoe_token', token)
        
        // Set authorization header for future requests
        api.defaults.headers.common['Authorization'] = `Basic ${token}`
        
        toast.success('Login successful!')
        
        // Redirect to dashboard
        window.location.href = '/dashboard'
      } else if (response.data.requiresTwoFactor) {
        // 2FA required
        setRequiresTwoFactor(true)
        toast.info('Please enter your 2FA code')
      } else {
        toast.error(response.data.message || 'Login failed')
      }
    } catch (error) {
      const errorMessage = error.response?.data?.message || 'Login failed'
      toast.error(errorMessage)
      
      // If 2FA is required but not provided yet
      if (error.response?.status === 400 && errorMessage.includes('two-factor')) {
        setRequiresTwoFactor(true)
      }
    } finally {
      setIsLoading(false)
    }
  }

  const handleQuickLogin = (username, password) => {
    setCredentials({ 
      username, 
      password, 
      totpCode: '', 
      backupCode: '' 
    })
    setRequiresTwoFactor(false)
    setUseBackupCode(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {/* Header */}
        <div className="text-center">
          <div className="mx-auto h-16 w-16 bg-primary-500 rounded-full flex items-center justify-center">
            <Stethoscope className="h-8 w-8 text-white" />
          </div>
          <h2 className="mt-6 text-3xl font-bold text-neutral-900">
            Smart Shoe Platform
          </h2>
          <p className="mt-2 text-sm text-neutral-600">
            Diabetic Neuropathy Monitoring System
          </p>
        </div>

        {/* Login Form */}
        <div className="medical-card">
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="username" className="form-label">
                Username
              </label>
              <input
                id="username"
                name="username"
                type="text"
                required
                className="form-input"
                placeholder="Enter your username"
                value={credentials.username}
                onChange={(e) => setCredentials(prev => ({
                  ...prev,
                  username: e.target.value
                }))}
              />
            </div>

            <div>
              <label htmlFor="password" className="form-label">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  required
                  className="form-input pr-10"
                  placeholder="Enter your password"
                  value={credentials.password}
                  onChange={(e) => setCredentials(prev => ({
                    ...prev,
                    password: e.target.value
                  }))}
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4 text-neutral-400" />
                  ) : (
                    <Eye className="h-4 w-4 text-neutral-400" />
                  )}
                </button>
              </div>
            </div>

            {/* Two-Factor Authentication */}
            {requiresTwoFactor && (
              <div className="space-y-4">
                <div className="border-t border-neutral-200 pt-4">
                  <div className="flex items-center justify-center space-x-2 mb-4">
                    <Smartphone className="h-5 w-5 text-primary-500" />
                    <h3 className="text-lg font-medium text-neutral-900">
                      Two-Factor Authentication
                    </h3>
                  </div>
                  
                  <div className="flex justify-center space-x-4 mb-4">
                    <button
                      type="button"
                      onClick={() => setUseBackupCode(false)}
                      className={`px-3 py-2 text-sm rounded-md ${
                        !useBackupCode 
                          ? 'bg-primary-100 text-primary-700 border border-primary-200' 
                          : 'text-neutral-600 hover:text-neutral-900'
                      }`}
                    >
                      Authenticator App
                    </button>
                    <button
                      type="button"
                      onClick={() => setUseBackupCode(true)}
                      className={`px-3 py-2 text-sm rounded-md ${
                        useBackupCode 
                          ? 'bg-primary-100 text-primary-700 border border-primary-200' 
                          : 'text-neutral-600 hover:text-neutral-900'
                      }`}
                    >
                      Backup Code
                    </button>
                  </div>
                  
                  {!useBackupCode ? (
                    <div>
                      <label htmlFor="totpCode" className="form-label">
                        Authentication Code
                      </label>
                      <input
                        id="totpCode"
                        name="totpCode"
                        type="text"
                        className="form-input text-center text-lg font-mono tracking-widest"
                        placeholder="000000"
                        value={credentials.totpCode}
                        onChange={(e) => setCredentials(prev => ({
                          ...prev,
                          totpCode: e.target.value.replace(/\D/g, '').slice(0, 6)
                        }))}
                        maxLength={6}
                      />
                      <p className="text-xs text-neutral-500 mt-1">
                        Enter the 6-digit code from your authenticator app
                      </p>
                    </div>
                  ) : (
                    <div>
                      <label htmlFor="backupCode" className="form-label">
                        Backup Code
                      </label>
                      <input
                        id="backupCode"
                        name="backupCode"
                        type="text"
                        className="form-input text-center font-mono"
                        placeholder="Enter backup code"
                        value={credentials.backupCode}
                        onChange={(e) => setCredentials(prev => ({
                          ...prev,
                          backupCode: e.target.value.replace(/\D/g, '').slice(0, 8)
                        }))}
                        maxLength={8}
                      />
                      <p className="text-xs text-neutral-500 mt-1">
                        Enter one of your 8-digit backup codes
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-500 hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <LoadingSpinner size="sm" color="neutral" />
              ) : (
                'Sign In'
              )}
            </button>
          </form>
        </div>

        {/* Quick Login Options */}
        <div className="medical-card">
          <h3 className="text-lg font-medium text-neutral-900 mb-4">
            Quick Login (Demo Credentials)
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => handleQuickLogin('admin', 'admin123')}
              className="flex items-center justify-center px-3 py-2 border border-neutral-300 rounded-md text-sm font-medium text-neutral-700 bg-white hover:bg-neutral-50 transition-colors"
            >
              <Shield className="h-4 w-4 mr-2 text-primary-500" />
              Admin
            </button>
            <button
              onClick={() => handleQuickLogin('doctor', 'doctor123')}
              className="flex items-center justify-center px-3 py-2 border border-neutral-300 rounded-md text-sm font-medium text-neutral-700 bg-white hover:bg-neutral-50 transition-colors"
            >
              <Stethoscope className="h-4 w-4 mr-2 text-secondary-500" />
              Doctor
            </button>
            <button
              onClick={() => handleQuickLogin('patient', 'patient123')}
              className="flex items-center justify-center px-3 py-2 border border-neutral-300 rounded-md text-sm font-medium text-neutral-700 bg-white hover:bg-neutral-50 transition-colors"
            >
              <User className="h-4 w-4 mr-2 text-success" />
              Patient
            </button>
            <button
              onClick={() => handleQuickLogin('demo', 'demo')}
              className="flex items-center justify-center px-3 py-2 border border-neutral-300 rounded-md text-sm font-medium text-neutral-700 bg-white hover:bg-neutral-50 transition-colors"
            >
              <Zap className="h-4 w-4 mr-2 text-warning" />
              Demo
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center">
          <p className="text-xs text-neutral-500">
            Advanced diabetic monitoring through smart shoe technology
          </p>
        </div>
      </div>
    </div>
  )
}

export default Login