import { useState, useEffect, Suspense } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from 'react-query'
import { ReactQueryDevtools } from 'react-query/devtools'
import { HelmetProvider } from 'react-helmet-async'
import { ErrorBoundary } from 'react-error-boundary'
import { Toaster } from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'

// Context providers
import { AuthProvider } from '@contexts/AuthContext'
import { TestSessionProvider } from '@contexts/TestSessionContext'
import { RoleProvider } from '@contexts/RoleContext'
import { ThemeProvider } from '@contexts/ThemeContext'
import { NotificationProvider } from '@contexts/NotificationContext'
import { WebSocketProvider } from '@contexts/WebSocketContext'

// Components
import ProtectedRoute from '@components/ProtectedRoute'
import Layout from '@components/layout/Layout'
import LoadingSpinner from '@components/common/LoadingSpinner'
import ErrorFallback from '@components/common/ErrorFallback'
import PWAInstallPrompt from '@components/common/PWAInstallPrompt'
import NetworkStatus from '@components/common/NetworkStatus'

// Lazy-loaded pages for better performance
import { lazy } from 'react'

// Authentication pages
const Login = lazy(() => import('@pages/Auth/Login'))
const Signup = lazy(() => import('@pages/Auth/Signup'))
const ForgotPassword = lazy(() => import('@pages/Auth/ForgotPassword'))
const ResetPassword = lazy(() => import('@pages/Auth/ResetPassword'))
const TwoFactorAuth = lazy(() => import('@pages/Auth/TwoFactorAuth'))

// Main application pages
const Dashboard = lazy(() => import('@pages/Dashboard/Dashboard'))
const PatientProfile = lazy(() => import('@pages/PatientProfile/PatientProfile'))
const PatientManagement = lazy(() => import('@pages/PatientManagement/PatientManagement'))
const TestResults = lazy(() => import('@pages/TestResults/TestResults'))
const TestSession = lazy(() => import('@pages/TestSession/TestSession'))
const NeuropathyMonitor = lazy(() => import('@pages/NeuropathyMonitor/NeuropathyMonitor'))
const DeviceManagement = lazy(() => import('@pages/DeviceManagement/DeviceManagement'))
const MLPredictions = lazy(() => import('@pages/MLPredictions/MLPredictions'))
const Analytics = lazy(() => import('@pages/Analytics/Analytics'))
const Reports = lazy(() => import('@pages/Reports/Reports'))
const Alerts = lazy(() => import('@pages/Alerts/Alerts'))
const Settings = lazy(() => import('@pages/Settings/Settings'))
const UserManagement = lazy(() => import('@pages/UserManagement/UserManagement'))
const SystemHealth = lazy(() => import('@pages/SystemHealth/SystemHealth'))
const AuditLogs = lazy(() => import('@pages/AuditLogs/AuditLogs'))

// Specialized pages
const MedicalHistory = lazy(() => import('@pages/MedicalHistory/MedicalHistory'))
const Medications = lazy(() => import('@pages/Medications/Medications'))
const Appointments = lazy(() => import('@pages/Appointments/Appointments'))
const Telemedicine = lazy(() => import('@pages/Telemedicine/Telemedicine'))
const Research = lazy(() => import('@pages/Research/Research'))
const Compliance = lazy(() => import('@pages/Compliance/Compliance'))

// Error pages
const NotFound = lazy(() => import('@pages/Error/NotFound'))
const ServerError = lazy(() => import('@pages/Error/ServerError'))
const Maintenance = lazy(() => import('@pages/Error/Maintenance'))

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
    },
  },
})

// Route animation variants
const pageVariants = {
  initial: { opacity: 0, y: 20 },
  in: { opacity: 1, y: 0 },
  out: { opacity: 0, y: -20 }
}

const pageTransition = {
  type: 'tween',
  ease: 'anticipate',
  duration: 0.4
}

// Main App component
function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [maintenanceMode, setMaintenanceMode] = useState(false)
  const location = useLocation()

  // Handle network status changes
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  // Check for maintenance mode
  useEffect(() => {
    const checkMaintenanceMode = async () => {
      try {
        const response = await fetch('/api/admin/maintenance-status')
        const data = await response.json()
        setMaintenanceMode(data.maintenance)
      } catch (error) {
        console.warn('Could not check maintenance status:', error)
      }
    }

    checkMaintenanceMode()
    const interval = setInterval(checkMaintenanceMode, 30000) // Check every 30 seconds

    return () => clearInterval(interval)
  }, [])

  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev)
  }

  // Show maintenance page if in maintenance mode
  if (maintenanceMode) {
    return (
      <Suspense fallback={<LoadingSpinner />}>
        <Maintenance />
      </Suspense>
    )
  }

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <HelmetProvider>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider>
            <AuthProvider>
              <RoleProvider>
                <NotificationProvider>
                  <WebSocketProvider>
                    <TestSessionProvider>
                      <div className="App">
                        {/* Network status indicator */}
                        <NetworkStatus isOnline={isOnline} />
                        
                        {/* PWA install prompt */}
                        <PWAInstallPrompt />
                        
                        {/* Toast notifications */}
                        <Toaster
                          position="top-right"
                          toastOptions={{
                            duration: 4000,
                            style: {
                              background: '#363636',
                              color: '#fff',
                            },
                            success: {
                              style: {
                                background: '#10B981',
                              },
                            },
                            error: {
                              style: {
                                background: '#EF4444',
                              },
                            },
                          }}
                        />

                        {/* Main routing */}
                        <AnimatePresence mode="wait">
                          <Routes location={location} key={location.pathname}>
                            {/* Public routes */}
                            <Route 
                              path="/login" 
                              element={
                                <motion.div
                                  initial="initial"
                                  animate="in"
                                  exit="out"
                                  variants={pageVariants}
                                  transition={pageTransition}
                                >
                                  <Suspense fallback={<LoadingSpinner />}>
                                    <Login />
                                  </Suspense>
                                </motion.div>
                              } 
                            />
                            <Route 
                              path="/signup" 
                              element={
                                <motion.div
                                  initial="initial"
                                  animate="in"
                                  exit="out"
                                  variants={pageVariants}
                                  transition={pageTransition}
                                >
                                  <Suspense fallback={<LoadingSpinner />}>
                                    <Signup />
                                  </Suspense>
                                </motion.div>
                              } 
                            />
                            <Route 
                              path="/forgot-password" 
                              element={
                                <motion.div
                                  initial="initial"
                                  animate="in"
                                  exit="out"
                                  variants={pageVariants}
                                  transition={pageTransition}
                                >
                                  <Suspense fallback={<LoadingSpinner />}>
                                    <ForgotPassword />
                                  </Suspense>
                                </motion.div>
                              } 
                            />
                            <Route 
                              path="/reset-password" 
                              element={
                                <motion.div
                                  initial="initial"
                                  animate="in"
                                  exit="out"
                                  variants={pageVariants}
                                  transition={pageTransition}
                                >
                                  <Suspense fallback={<LoadingSpinner />}>
                                    <ResetPassword />
                                  </Suspense>
                                </motion.div>
                              } 
                            />
                            <Route 
                              path="/2fa" 
                              element={
                                <motion.div
                                  initial="initial"
                                  animate="in"
                                  exit="out"
                                  variants={pageVariants}
                                  transition={pageTransition}
                                >
                                  <Suspense fallback={<LoadingSpinner />}>
                                    <TwoFactorAuth />
                                  </Suspense>
                                </motion.div>
                              } 
                            />

                            {/* Protected routes */}
                            <Route element={
                              <ProtectedRoute>
                                <Layout sidebarOpen={sidebarOpen} toggleSidebar={toggleSidebar} />
                              </ProtectedRoute>
                            }>
                              {/* Dashboard */}
                              <Route path="/" element={
                                <motion.div
                                  initial="initial"
                                  animate="in"
                                  exit="out"
                                  variants={pageVariants}
                                  transition={pageTransition}
                                >
                                  <Suspense fallback={<LoadingSpinner />}>
                                    <Dashboard />
                                  </Suspense>
                                </motion.div>
                              } />

                              {/* Patient Management */}
                              <Route path="/patients" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <PatientManagement />
                                </Suspense>
                              } />
                              <Route path="/patients/:id" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <PatientProfile />
                                </Suspense>
                              } />

                              {/* Testing & Monitoring */}
                              <Route path="/test-sessions" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <TestSession />
                                </Suspense>
                              } />
                              <Route path="/test-results" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <TestResults />
                                </Suspense>
                              } />
                              <Route path="/neuropathy-monitor" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <NeuropathyMonitor />
                                </Suspense>
                              } />

                              {/* Device Management */}
                              <Route path="/devices" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <DeviceManagement />
                                </Suspense>
                              } />

                              {/* ML & Analytics */}
                              <Route path="/ml-predictions" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <MLPredictions />
                                </Suspense>
                              } />
                              <Route path="/analytics" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Analytics />
                                </Suspense>
                              } />

                              {/* Medical Data */}
                              <Route path="/medical-history" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <MedicalHistory />
                                </Suspense>
                              } />
                              <Route path="/medications" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Medications />
                                </Suspense>
                              } />

                              {/* Appointments & Telemedicine */}
                              <Route path="/appointments" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Appointments />
                                </Suspense>
                              } />
                              <Route path="/telemedicine" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Telemedicine />
                                </Suspense>
                              } />

                              {/* Alerts & Notifications */}
                              <Route path="/alerts" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Alerts />
                                </Suspense>
                              } />

                              {/* Reports */}
                              <Route path="/reports" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Reports />
                                </Suspense>
                              } />

                              {/* Administration */}
                              <Route path="/user-management" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <UserManagement />
                                </Suspense>
                              } />
                              <Route path="/system-health" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <SystemHealth />
                                </Suspense>
                              } />
                              <Route path="/audit-logs" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <AuditLogs />
                                </Suspense>
                              } />

                              {/* Research & Compliance */}
                              <Route path="/research" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Research />
                                </Suspense>
                              } />
                              <Route path="/compliance" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Compliance />
                                </Suspense>
                              } />

                              {/* Settings */}
                              <Route path="/settings" element={
                                <Suspense fallback={<LoadingSpinner />}>
                                  <Settings />
                                </Suspense>
                              } />

                              {/* Legacy routes for backward compatibility */}
                              <Route path="/patient-profile" element={<Navigate to="/patients" replace />} />
                            </Route>

                            {/* Error pages */}
                            <Route path="/500" element={
                              <Suspense fallback={<LoadingSpinner />}>
                                <ServerError />
                              </Suspense>
                            } />
                            <Route path="/404" element={
                              <Suspense fallback={<LoadingSpinner />}>
                                <NotFound />
                              </Suspense>
                            } />

                            {/* Catch all route */}
                            <Route path="*" element={<Navigate to="/404" replace />} />
                          </Routes>
                        </AnimatePresence>
                      </div>
                    </TestSessionProvider>
                  </WebSocketProvider>
                </NotificationProvider>
              </RoleProvider>
            </AuthProvider>
          </ThemeProvider>
          
          {/* React Query Dev Tools (only in development) */}
          {process.env.NODE_ENV === 'development' && (
            <ReactQueryDevtools initialIsOpen={false} />
          )}
        </QueryClientProvider>
      </HelmetProvider>
    </ErrorBoundary>
  )
}

export default App