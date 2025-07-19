import { createContext, useContext, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '@services/api'
import { useNotifications } from './NotificationContext'

const AuthContext = createContext({})

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const navigate = useNavigate()
  const { showSuccess, showError } = useNotifications()

  // Check if user is logged in on app start
  useEffect(() => {
    checkAuthStatus()
  }, [])

  const checkAuthStatus = async () => {
    try {
      const token = localStorage.getItem('auth_token')
      if (!token) {
        setLoading(false)
        return
      }

      // Verify token with backend
      const response = await api.auth.verifyToken()
      if (response.data.valid) {
        setUser(response.data.user)
        setIsAuthenticated(true)
      } else {
        // Token is invalid, clear it
        logout()
      }
    } catch (error) {
      console.error('Auth check failed:', error)
      logout()
    } finally {
      setLoading(false)
    }
  }

  const login = async (email, password, rememberMe = false) => {
    try {
      setLoading(true)
      const response = await api.auth.login({
        email,
        password,
        rememberMe
      })

      const { user: userData, token, refreshToken, requiresTwoFactor } = response.data

      if (requiresTwoFactor) {
        // Store temporary token for 2FA verification
        localStorage.setItem('temp_token', token)
        return { requiresTwoFactor: true, tempToken: token }
      }

      // Store tokens
      localStorage.setItem('auth_token', token)
      if (refreshToken) {
        localStorage.setItem('refresh_token', refreshToken)
      }

      // Store user data
      setUser(userData)
      setIsAuthenticated(true)

      showSuccess(`Welcome back, ${userData.firstName}!`)
      
      // Redirect based on role
      const redirectPath = getRedirectPath(userData.role)
      navigate(redirectPath)

      return { success: true, user: userData }

    } catch (error) {
      const errorMessage = error.response?.data?.message || 'Login failed. Please check your credentials.'
      showError(errorMessage)
      throw error
    } finally {
      setLoading(false)
    }
  }

  const verifyTwoFactor = async (code, tempToken) => {
    try {
      setLoading(true)
      const response = await api.auth.verifyTwoFactor({
        code,
        tempToken
      })

      const { user: userData, token, refreshToken } = response.data

      // Store tokens
      localStorage.setItem('auth_token', token)
      if (refreshToken) {
        localStorage.setItem('refresh_token', refreshToken)
      }
      
      // Remove temp token
      localStorage.removeItem('temp_token')

      // Store user data
      setUser(userData)
      setIsAuthenticated(true)

      showSuccess('Two-factor authentication successful!')
      
      // Redirect based on role
      const redirectPath = getRedirectPath(userData.role)
      navigate(redirectPath)

      return { success: true, user: userData }

    } catch (error) {
      const errorMessage = error.response?.data?.message || 'Invalid verification code.'
      showError(errorMessage)
      throw error
    } finally {
      setLoading(false)
    }
  }

  const register = async (userData) => {
    try {
      setLoading(true)
      const response = await api.auth.register(userData)

      const { user: newUser, token, refreshToken, requiresEmailVerification } = response.data

      if (requiresEmailVerification) {
        showSuccess('Registration successful! Please check your email to verify your account.')
        return { requiresEmailVerification: true, email: userData.email }
      }

      // Store tokens
      localStorage.setItem('auth_token', token)
      if (refreshToken) {
        localStorage.setItem('refresh_token', refreshToken)
      }

      // Store user data
      setUser(newUser)
      setIsAuthenticated(true)

      showSuccess(`Welcome to Smart Shoe Monitor, ${newUser.firstName}!`)
      
      // Redirect to onboarding or dashboard
      const redirectPath = newUser.isFirstLogin ? '/onboarding' : getRedirectPath(newUser.role)
      navigate(redirectPath)

      return { success: true, user: newUser }

    } catch (error) {
      const errorMessage = error.response?.data?.message || 'Registration failed. Please try again.'
      showError(errorMessage)
      throw error
    } finally {
      setLoading(false)
    }
  }

  const logout = async (showMessage = true) => {
    try {
      // Call logout endpoint to invalidate token on server
      const token = localStorage.getItem('auth_token')
      if (token) {
        await api.auth.logout()
      }
    } catch (error) {
      console.error('Logout API call failed:', error)
    } finally {
      // Clear local storage
      localStorage.removeItem('auth_token')
      localStorage.removeItem('refresh_token')
      localStorage.removeItem('temp_token')
      localStorage.removeItem('user_preferences')

      // Clear state
      setUser(null)
      setIsAuthenticated(false)

      if (showMessage) {
        showSuccess('You have been logged out successfully.')
      }

      // Redirect to login
      navigate('/login')
    }
  }

  const forgotPassword = async (email) => {
    try {
      setLoading(true)
      await api.auth.forgotPassword({ email })
      showSuccess('Password reset link sent to your email.')
      return { success: true }
    } catch (error) {
      const errorMessage = error.response?.data?.message || 'Failed to send reset email.'
      showError(errorMessage)
      throw error
    } finally {
      setLoading(false)
    }
  }

  const resetPassword = async (token, newPassword) => {
    try {
      setLoading(true)
      await api.auth.resetPassword({ token, newPassword })
      showSuccess('Password reset successfully. You can now log in with your new password.')
      navigate('/login')
      return { success: true }
    } catch (error) {
      const errorMessage = error.response?.data?.message || 'Failed to reset password.'
      showError(errorMessage)
      throw error
    } finally {
      setLoading(false)
    }
  }

  const updateProfile = async (updates) => {
    try {
      setLoading(true)
      const response = await api.auth.updateProfile(updates)
      const updatedUser = response.data.user
      
      setUser(updatedUser)
      showSuccess('Profile updated successfully.')
      
      return { success: true, user: updatedUser }
    } catch (error) {
      const errorMessage = error.response?.data?.message || 'Failed to update profile.'
      showError(errorMessage)
      throw error
    } finally {
      setLoading(false)
    }
  }

  const changePassword = async (currentPassword, newPassword) => {
    try {
      setLoading(true)
      await api.auth.changePassword({ currentPassword, newPassword })
      showSuccess('Password changed successfully.')
      return { success: true }
    } catch (error) {
      const errorMessage = error.response?.data?.message || 'Failed to change password.'
      showError(errorMessage)
      throw error
    } finally {
      setLoading(false)
    }
  }

  const refreshToken = async () => {
    try {
      const refreshToken = localStorage.getItem('refresh_token')
      if (!refreshToken) {
        throw new Error('No refresh token available')
      }

      const response = await api.auth.refreshToken({ refreshToken })
      const { token: newToken, refreshToken: newRefreshToken } = response.data

      localStorage.setItem('auth_token', newToken)
      if (newRefreshToken) {
        localStorage.setItem('refresh_token', newRefreshToken)
      }

      return newToken
    } catch (error) {
      console.error('Token refresh failed:', error)
      logout(false)
      throw error
    }
  }

  // Helper function to get redirect path based on role
  const getRedirectPath = (role) => {
    switch (role) {
      case 'ADMIN':
        return '/system-health'
      case 'PROVIDER':
        return '/patients'
      case 'CAREGIVER':
        return '/dashboard'
      case 'PATIENT':
      default:
        return '/dashboard'
    }
  }

  // Check if user has specific role
  const hasRole = (role) => {
    return user?.role === role
  }

  // Check if user has any of the specified roles
  const hasAnyRole = (roles) => {
    return roles.includes(user?.role)
  }

  // Check if user has permission for specific action
  const hasPermission = (permission) => {
    return user?.permissions?.includes(permission) || user?.role === 'ADMIN'
  }

  const value = {
    // State
    user,
    loading,
    isAuthenticated,
    
    // Actions
    login,
    register,
    logout,
    verifyTwoFactor,
    forgotPassword,
    resetPassword,
    updateProfile,
    changePassword,
    refreshToken,
    checkAuthStatus,
    
    // Utilities
    hasRole,
    hasAnyRole,
    hasPermission,
    getRedirectPath
  }

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}