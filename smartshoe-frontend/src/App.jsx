import React, { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext'
import { NotificationProvider } from './contexts/NotificationContext'
import { WebSocketProvider } from './contexts/WebSocketContext'
import { mobileServices } from './services/mobileServices'
import Layout from './components/Layout/Layout'
import ProtectedRoute from './components/auth/ProtectedRoute'
import Login from './pages/auth/Login'
import Dashboard from './pages/Dashboard'
import PatientManagement from './pages/PatientManagement'
import DeviceManagement from './pages/DeviceManagement'
import MedicalReadings from './pages/MedicalReadings'
import NeuropathyTesting from './pages/NeuropathyTesting'
import PatientProfile from './pages/PatientProfile'
import Settings from './pages/Settings'
import MLTesting from './pages/MLTesting'
import LoadingSpinner from './components/common/LoadingSpinner'

function App() {
  const { user, isLoading } = useAuth()

  // Initialize mobile services
  useEffect(() => {
    const initMobile = async () => {
      try {
        await mobileServices.initialize()
        console.log('Mobile services initialized')
        
        // Schedule a welcome notification for first-time users
        if (mobileServices.isNative && user) {
          await mobileServices.scheduleLocalNotification(
            'Welcome to Smart Shoe',
            'Your diabetic monitoring companion is ready to use!',
            { type: 'welcome', userId: user.id }
          )
        }
      } catch (error) {
        console.error('Failed to initialize mobile services:', error)
      }
    }

    initMobile()
  }, [user])

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-50">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (!user) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    )
  }

  return (
    <NotificationProvider>
      <WebSocketProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            
            {/* Admin & Provider only routes */}
            <Route 
              path="/patients" 
              element={
                <ProtectedRoute requiredRole="PROVIDER">
                  <PatientManagement />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/patients/:id" 
              element={
                <ProtectedRoute requiredRole="PROVIDER">
                  <PatientProfile />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/devices" 
              element={
                <ProtectedRoute requiredRole="PROVIDER">
                  <DeviceManagement />
                </ProtectedRoute>
              } 
            />
            
            {/* All authenticated users can access */}
            <Route path="/medical-readings" element={<MedicalReadings />} />
            <Route path="/neuropathy-testing" element={<NeuropathyTesting />} />
            <Route path="/settings" element={<Settings />} />
            
            {/* ML Testing - Provider only */}
            <Route 
              path="/ml-testing" 
              element={
                <ProtectedRoute requiredRole="PROVIDER">
                  <MLTesting />
                </ProtectedRoute>
              } 
            />
            
            <Route path="/login" element={<Navigate to="/dashboard" replace />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Layout>
      </WebSocketProvider>
    </NotificationProvider>
  )
}

export default App