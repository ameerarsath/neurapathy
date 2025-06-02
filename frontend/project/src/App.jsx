import { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import Layout from './components/layout/Layout'
import Login from './pages/Auth/Login'
import Signup from './pages/Auth/Signup'
import ForgotPassword from './pages/Auth/ForgotPassword'
import Dashboard from './pages/Dashboard/Dashboard'
import PatientProfile from './pages/PatientProfile/PatientProfile'
import TestResults from './pages/TestResults/TestResults'
import NeuropathyMonitor from './pages/NeuropathyMonitor/NeuropathyMonitor'
import Settings from './pages/Settings/Settings'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev)
  }

  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        
        <Route element={
          <ProtectedRoute>
            <Layout sidebarOpen={sidebarOpen} toggleSidebar={toggleSidebar} />
          </ProtectedRoute>
        }>
          <Route path="/" element={<Dashboard />} />
          <Route path="/patient-profile" element={<PatientProfile />} />
          <Route path="/test-results" element={<TestResults />} />
          <Route path="/neuropathy-monitor" element={<NeuropathyMonitor />} />
          <Route path="/settings" element={<Settings />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  )
}

export default App