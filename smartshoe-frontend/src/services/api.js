import axios from 'axios'
import toast from 'react-hot-toast'

// Create axios instance
const api = axios.create({
  baseURL: 'http://localhost:8080',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('smartshoe_token')
    if (token) {
      config.headers.Authorization = `Basic ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('smartshoe_token')
      delete api.defaults.headers.common['Authorization']
      window.location.href = '/login'
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Please try again later.')
    }
    return Promise.reject(error)
  }
)

// API endpoints
export const smartShoeAPI = {
  // Authentication
  auth: {
    login: (credentials) => api.post('/api/auth/login', credentials),
    logout: () => api.post('/api/auth/logout'),
    getProfile: () => api.get('/api/auth/profile'),
    changePassword: (data) => api.post('/api/auth/change-password', data),
    
    // Two-Factor Authentication
    getTwoFactorStatus: () => api.get('/api/auth/2fa/status'),
    enableTwoFactor: () => api.post('/api/auth/2fa/enable'),
    verifyTwoFactorSetup: (data) => api.post('/api/auth/2fa/verify-setup', data),
    disableTwoFactor: (data) => api.post('/api/auth/2fa/disable', data),
    regenerateBackupCodes: () => api.post('/api/auth/2fa/regenerate-backup-codes'),
  },

  // Public endpoints
  public: {
    health: () => api.get('/api/health'),
    status: () => api.get('/api/status'),
    test: () => api.get('/api/test'),
    credentials: () => api.get('/api/credentials'),
    endpoints: () => api.get('/api/endpoints'),
  },

  // Dashboard data
  dashboard: {
    getData: (userId) => api.get(`/api/dashboard/${userId}`),
    getStatistics: () => api.get('/api/dashboard/statistics'),
    getRecentActivity: () => api.get('/api/dashboard/recent-activity'),
  },

  // Patient management
  patients: {
    getAll: (params = {}) => api.get('/api/patients', { params }),
    getById: (id) => api.get(`/api/patients/${id}`),
    create: (data) => api.post('/api/patients', data),
    update: (id, data) => api.put(`/api/patients/${id}`, data),
    delete: (id) => api.delete(`/api/patients/${id}`),
    search: (query) => api.get('/api/patients/search', { params: { name: query } }),
    getByDiabetesType: (type) => api.get(`/api/patients/diabetes-type/${type}`),
    getByAgeRange: (minAge, maxAge) => api.get('/api/patients/age-range', { 
      params: { minAge, maxAge } 
    }),
    getStatistics: () => api.get('/api/patients/statistics'),
  },

  // Device management
  devices: {
    getAll: (params = {}) => api.get('/api/devices', { params }),
    getById: (id) => api.get(`/api/devices/${id}`),
    create: (data) => api.post('/api/devices', data),
    update: (id, data) => api.put(`/api/devices/${id}`, data),
    delete: (id) => api.delete(`/api/devices/${id}`),
    assignToPatient: (deviceId, patientId) => 
      api.post(`/api/devices/${deviceId}/assign/${patientId}`),
    unassign: (deviceId) => api.post(`/api/devices/${deviceId}/unassign`),
    calibrate: (deviceId) => api.post(`/api/devices/${deviceId}/calibrate`),
    updateBattery: (deviceId, level) => 
      api.put(`/api/devices/${deviceId}/battery`, { batteryLevel: level }),
    updateSync: (deviceId) => api.put(`/api/devices/${deviceId}/sync`),
    getByPatient: (patientId) => api.get(`/api/devices/patient/${patientId}`),
    getByStatus: (status) => api.get(`/api/devices/status/${status}`),
    getLowBattery: () => api.get('/api/devices/low-battery'),
    getOffline: () => api.get('/api/devices/offline'),
    getStatistics: () => api.get('/api/devices/statistics'),
  },

  // Medical readings
  medicalReadings: {
    getAll: (params = {}) => api.get('/api/medical-readings', { params }),
    getById: (id) => api.get(`/api/medical-readings/${id}`),
    create: (data) => api.post('/api/medical-readings', data),
    update: (id, data) => api.put(`/api/medical-readings/${id}`, data),
    delete: (id) => api.delete(`/api/medical-readings/${id}`),
    getByPatient: (patientId, params = {}) => 
      api.get(`/api/medical-readings/patient/${patientId}`, { params }),
    getByDevice: (deviceId) => api.get(`/api/medical-readings/device/${deviceId}`),
    getByType: (type) => api.get(`/api/medical-readings/type/${type}`),
    getByDateRange: (startDate, endDate) => 
      api.get('/api/medical-readings/date-range', { 
        params: { startDate, endDate } 
      }),
    getAbnormal: () => api.get('/api/medical-readings/abnormal'),
    getCritical: () => api.get('/api/medical-readings/critical'),
    getBaseline: (patientId) => 
      api.get(`/api/medical-readings/baseline/${patientId}`),
    markAsBaseline: (id) => 
      api.post(`/api/medical-readings/${id}/baseline`),
    getStatistics: () => api.get('/api/medical-readings/statistics'),
    
    // Export endpoints
    exportToCSV: () => api.get('/api/medical-readings/export/csv', { responseType: 'blob' }),
    exportToExcel: () => api.get('/api/medical-readings/export/excel', { responseType: 'blob' }),
    exportToPDF: () => api.get('/api/medical-readings/export/pdf', { responseType: 'blob' }),
    exportPatientToCSV: (patientId) => 
      api.get(`/api/medical-readings/patient/${patientId}/export/csv`, { responseType: 'blob' }),
    exportPatientToPDF: (patientId) => 
      api.get(`/api/medical-readings/patient/${patientId}/export/pdf`, { responseType: 'blob' }),
    exportDateRangeToCSV: (startDate, endDate) => 
      api.get('/api/medical-readings/export/csv/date-range', { 
        params: { startDate, endDate }, 
        responseType: 'blob' 
      }),
  },
}

export default api