import React from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../../contexts/AuthContext'

const ProtectedRoute = ({ children, requiredRole, fallbackPath = '/dashboard' }) => {
  const { user, canAccess, isLoading } = useAuth()

  // Show loading spinner while checking authentication
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  // If not authenticated, redirect to login
  if (!user) {
    return <Navigate to="/login" replace />
  }

  // If specific role required and user doesn't have access, redirect to fallback
  if (requiredRole && !canAccess(requiredRole)) {
    return <Navigate to={fallbackPath} replace />
  }

  // User has access, render the protected component
  return children
}

export default ProtectedRoute