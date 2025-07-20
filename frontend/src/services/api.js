import axios from 'axios'
import { toast } from 'react-hot-toast'

// API Configuration
const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080/api',
  mlBaseURL: import.meta.env.VITE_ML_API_BASE_URL || 'http://localhost:8080/api/ml',
  timeout: 30000,
  retryAttempts: 3,
  retryDelay: 1000
}

// Create axios instances
const apiClient = axios.create({
  baseURL: API_CONFIG.baseURL,
  timeout: API_CONFIG.timeout,
  headers: {
    'Content-Type': 'application/json',
    'X-Client-Version': '1.0.0',
    'X-Platform': 'web'
  }
})

const mlApiClient = axios.create({
  baseURL: API_CONFIG.mlBaseURL,
  timeout: API_CONFIG.timeout,
  headers: {
    'Content-Type': 'application/json',
    'X-Client-Version': '1.0.0'
  }
})

// Token management
const tokenManager = {
  getToken: () => localStorage.getItem('auth_token'),
  setToken: (token) => localStorage.setItem('auth_token', token),
  removeToken: () => localStorage.removeItem('auth_token'),
  getRefreshToken: () => localStorage.getItem('refresh_token'),
  setRefreshToken: (token) => localStorage.setItem('refresh_token', token),
  removeRefreshToken: () => localStorage.removeItem('refresh_token')
}

// Generate request ID for tracking
const generateRequestId = () => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2, 9)
}

// Request interceptor for authentication
apiClient.interceptors.request.use(
  (config) => {
    const token = tokenManager.getToken()
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    // Add request ID for tracking
    config.headers['X-Request-ID'] = generateRequestId()
    
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor for error handling and token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config
    
    // Handle 401 errors and token refresh
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true
      
      try {
        const refreshToken = tokenManager.getRefreshToken()
        if (refreshToken) {
          const response = await axios.post(`${API_CONFIG.baseURL}/auth/refresh`, {
            refreshToken
          })
          
          const { token: newToken } = response.data
          tokenManager.setToken(newToken)
          originalRequest.headers.Authorization = `Bearer ${newToken}`
          
          return apiClient(originalRequest)
        }
      } catch (refreshError) {
        // Refresh failed, redirect to login
        tokenManager.removeToken()
        tokenManager.removeRefreshToken()
        window.location.href = '/login'
        return Promise.reject(refreshError)
      }
    }
    
    // Handle other errors
    if (error.response?.status >= 500) {
      toast.error('Server error. Please try again later.')
    } else if (error.response?.status === 403) {
      toast.error('You do not have permission to perform this action.')
    }
    
    return Promise.reject(error)
  }
)

// API Endpoints
export const api = {
  // Authentication endpoints
  auth: {
    login: (credentials) => apiClient.post('/auth/login', credentials),
    register: (userData) => apiClient.post('/auth/register', userData),
    logout: () => apiClient.post('/auth/logout'),
    verifyToken: () => apiClient.get('/auth/verify'),
    verifyTwoFactor: (data) => apiClient.post('/auth/2fa/verify', data),
    forgotPassword: (data) => apiClient.post('/auth/forgot-password', data),
    resetPassword: (data) => apiClient.post('/auth/reset-password', data),
    changePassword: (data) => apiClient.put('/auth/change-password', data),
    updateProfile: (data) => apiClient.put('/auth/profile', data),
    refreshToken: (data) => apiClient.post('/auth/refresh', data),
    verifyEmail: (token) => apiClient.post(`/auth/verify-email?token=${token}`)
  },

  // Patient endpoints
  patient: {
    getPatients: (params) => apiClient.get('/patients', { params }),
    getPatient: (id) => apiClient.get(`/patients/${id}`),
    createPatient: (data) => apiClient.post('/patients', data),
    updatePatient: (id, data) => apiClient.put(`/patients/${id}`, data),
    deletePatient: (id) => apiClient.delete(`/patients/${id}`),
    getPatientStats: (id) => apiClient.get(`/patients/${id}/stats`),
    getPatientTimeline: (id) => apiClient.get(`/patients/${id}/timeline`)
  },

  // Device endpoints
  device: {
    getDevices: (params) => apiClient.get('/devices', { params }),
    getDevice: (id) => apiClient.get(`/devices/${id}`),
    registerDevice: (data) => apiClient.post('/devices/register', data),
    updateDevice: (id, data) => apiClient.put(`/devices/${id}`, data),
    deleteDevice: (id) => apiClient.delete(`/devices/${id}`),
    calibrateDevice: (id, data) => apiClient.post(`/devices/${id}/calibrate`, data),
    getDeviceStatus: (id) => apiClient.get(`/devices/${id}/status`),
    updateFirmware: (id, data) => apiClient.post(`/devices/${id}/firmware`, data)
  },

  // Test endpoints
  test: {
    getTestSessions: (params) => apiClient.get('/tests/sessions', { params }),
    getTestSession: (id) => apiClient.get(`/tests/sessions/${id}`),
    createTestSession: (data) => apiClient.post('/tests/sessions', data),
    updateTestSession: (id, data) => apiClient.put(`/tests/sessions/${id}`, data),
    getTestResults: (params) => apiClient.get('/tests/results', { params }),
    getTestResult: (id) => apiClient.get(`/tests/results/${id}`),
    createTestResult: (data) => apiClient.post('/tests/results', data),
    getBaselineReadings: (params) => apiClient.get('/tests/baseline', { params })
  },

  // Medical endpoints
  medical: {
    getMedicalHistory: (patientId) => apiClient.get(`/medical/history/${patientId}`),
    createMedicalRecord: (data) => apiClient.post('/medical/records', data),
    updateMedicalRecord: (id, data) => apiClient.put(`/medical/records/${id}`, data),
    getVitalSigns: (patientId) => apiClient.get(`/medical/vitals/${patientId}`),
    createVitalSigns: (data) => apiClient.post('/medical/vitals', data),
    getMedications: (patientId) => apiClient.get(`/medical/medications/${patientId}`),
    createMedication: (data) => apiClient.post('/medical/medications', data),
    updateMedication: (id, data) => apiClient.put(`/medical/medications/${id}`, data)
  },

  // Analytics endpoints
  analytics: {
    getPatientAnalytics: (patientId, params) => apiClient.get(`/analytics/patient/${patientId}`, { params }),
    getDetailedAnalytics: (patientId, params) => apiClient.get(`/analytics/detailed/${patientId}`, { params }),
    getProgressionAnalysis: (patientId, params) => apiClient.get(`/analytics/progression/${patientId}`, { params }),
    getRiskAssessment: (patientId) => apiClient.get(`/analytics/risk/${patientId}`),
    getComplianceMetrics: (patientId) => apiClient.get(`/analytics/compliance/${patientId}`)
  },

  // Dashboard endpoints
  dashboard: {
    getData: (userId) => apiClient.get(`/dashboard/${userId}`),
    getStatistics: () => apiClient.get('/dashboard/statistics'),
    getRecentActivity: () => apiClient.get('/dashboard/recent-activity')
  },

  // ML endpoints
  ml: {
    predictNeuropathyProgression: (patientId, data) => mlApiClient.post(`/predict/neuropathy/${patientId}`, data),
    predictGlucoseComplications: (patientId, data) => mlApiClient.post(`/predict/glucose/${patientId}`, data),
    detectAnomalies: (patientId, data) => mlApiClient.post(`/detect/anomalies/${patientId}`, data),
    getRiskStratification: (patientId, data) => mlApiClient.post(`/risk/stratification/${patientId}`, data),
    getPredictions: (patientId, params) => mlApiClient.get(`/predictions/${patientId}`, { params }),
    getModelMetrics: () => mlApiClient.get('/models/metrics')
  },

  // Alert endpoints
  alert: {
    getAlerts: (params) => apiClient.get('/alerts', { params }),
    getAlert: (id) => apiClient.get(`/alerts/${id}`),
    createAlert: (data) => apiClient.post('/alerts', data),
    updateAlert: (id, data) => apiClient.put(`/alerts/${id}`, data),
    deleteAlert: (id) => apiClient.delete(`/alerts/${id}`),
    markAsRead: (id) => apiClient.post(`/alerts/${id}/read`),
    getAlertConfiguration: (patientId) => apiClient.get(`/alerts/config/${patientId}`),
    updateAlertConfiguration: (patientId, data) => apiClient.put(`/alerts/config/${patientId}`, data)
  },

  // User management endpoints (Admin)
  admin: {
    getUsers: (params) => apiClient.get('/admin/users', { params }),
    getUser: (id) => apiClient.get(`/admin/users/${id}`),
    createUser: (data) => apiClient.post('/admin/users', data),
    updateUser: (id, data) => apiClient.put(`/admin/users/${id}`, data),
    deleteUser: (id) => apiClient.delete(`/admin/users/${id}`),
    getSystemHealth: () => apiClient.get('/admin/system/health'),
    getAuditLogs: (params) => apiClient.get('/admin/audit', { params }),
    getSystemStats: () => apiClient.get('/admin/stats')
  },

  // Appointment endpoints
  appointment: {
    getAppointments: (params) => apiClient.get('/appointments', { params }),
    getAppointment: (id) => apiClient.get(`/appointments/${id}`),
    createAppointment: (data) => apiClient.post('/appointments', data),
    updateAppointment: (id, data) => apiClient.put(`/appointments/${id}`, data),
    deleteAppointment: (id) => apiClient.delete(`/appointments/${id}`),
    getAvailableSlots: (providerId, date) => apiClient.get(`/appointments/slots/${providerId}`, { params: { date } })
  },

  // Report endpoints
  report: {
    generatePatientReport: (patientId, params) => apiClient.get(`/reports/patient/${patientId}`, { params, responseType: 'blob' }),
    generateProgressionReport: (patientId, params) => apiClient.get(`/reports/progression/${patientId}`, { params, responseType: 'blob' }),
    generateComplianceReport: (patientId, params) => apiClient.get(`/reports/compliance/${patientId}`, { params, responseType: 'blob' }),
    getReportHistory: (params) => apiClient.get('/reports/history', { params })
  }
}

// Export individual clients for advanced usage
export { apiClient, mlApiClient, tokenManager }