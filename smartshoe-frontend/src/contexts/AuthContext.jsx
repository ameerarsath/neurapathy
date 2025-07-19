import React, { createContext, useContext, useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import toast from 'react-hot-toast'
import api from '../services/api'

const AuthContext = createContext({})

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    const initializeAuth = async () => {
      const token = localStorage.getItem('smartshoe_token')
      
      if (token) {
        try {
          // Set the authorization header
          api.defaults.headers.common['Authorization'] = `Basic ${token}`
          
          // Verify token by making a test request
          await api.get('/api/health')
          
          // Decode the basic auth to get username
          let username;
          try {
            const decoded = atob(token)
            const parts = decoded.split(':')
            if (parts.length >= 2) {
              username = parts[0]
            } else {
              throw new Error('Invalid token format')
            }
          } catch (decodeError) {
            console.error('Token decode error:', decodeError)
            localStorage.removeItem('smartshoe_token')
            delete api.defaults.headers.common['Authorization']
            return
          }
          
          setUser({
            id: getUserId(username),
            username,
            role: getUserRole(username),
            token
          })
        } catch (error) {
          console.error('Token validation failed:', error)
          localStorage.removeItem('smartshoe_token')
          delete api.defaults.headers.common['Authorization']
        }
      }
      
      setIsLoading(false)
    }

    initializeAuth()
  }, [])

  const getUserRole = (username) => {
    // Map usernames to roles based on backend credentials
    const roleMap = {
      'admin': 'ADMIN',
      'doctor': 'PROVIDER', 
      'patient': 'PATIENT',
      'demo': 'USER'
    }
    return roleMap[username] || 'USER'
  }

  const getUserId = (username) => {
    // Map usernames to Patient table IDs (for medical data access)
    // Note: User auth table and Patient domain table are separate
    const idMap = {
      'admin': null,     // Admin doesn't map to a specific patient
      'doctor': null,    // Doctor doesn't map to a specific patient
      'patient': 1,      // Patient user maps to Patient ID=1 (John Doe)
      'demo': 1
    }
    return idMap[username] || null
  }

  const login = async (username, password) => {
    try {
      setIsLoading(true)
      
      // Create basic auth token
      const token = btoa(`${username}:${password}`)
      
      // Test the credentials with a secured endpoint that requires authentication
      const response = await api.get('/api/patients', {
        headers: {
          'Authorization': `Basic ${token}`
        }
      })
      
      if (response.status === 200) {
        // Set authorization header for future requests
        api.defaults.headers.common['Authorization'] = `Basic ${token}`
        
        // Store token
        localStorage.setItem('smartshoe_token', token)
        
        // Set user state
        const userData = {
          id: getUserId(username),
          username,
          role: getUserRole(username),
          token
        }
        setUser(userData)
        
        toast.success(`Welcome back, ${username}!`)
        
        // Redirect to intended location or dashboard
        const intendedPath = location.state?.from?.pathname || '/dashboard'
        navigate(intendedPath, { replace: true })
        
        return { success: true }
      }
    } catch (error) {
      console.error('Login error:', error)
      
      let errorMessage = 'Login failed. Please check your credentials.'
      if (error.response?.status === 401) {
        errorMessage = 'Invalid username or password.'
      } else if (error.response?.status >= 500) {
        errorMessage = 'Server error. Please try again later.'
      }
      
      toast.error(errorMessage)
      return { success: false, error: errorMessage }
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem('smartshoe_token')
    delete api.defaults.headers.common['Authorization']
    toast.success('Logged out successfully')
    navigate('/login')
  }

  const hasRole = (role) => {
    if (!user) return false
    
    const roleHierarchy = {
      'ADMIN': ['ADMIN', 'PROVIDER', 'PATIENT', 'USER'],
      'PROVIDER': ['PROVIDER', 'PATIENT', 'USER'],
      'PATIENT': ['PATIENT', 'USER'],
      'USER': ['USER']
    }
    
    return roleHierarchy[user.role]?.includes(role) || false
  }

  const canAccess = (requiredRole) => {
    return hasRole(requiredRole)
  }

  const value = {
    user,
    isLoading,
    login,
    logout,
    hasRole,
    canAccess,
    isAuthenticated: !!user
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}