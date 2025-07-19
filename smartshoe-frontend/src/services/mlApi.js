import axios from 'axios'

// Create a completely isolated axios instance for ML API calls (no auth)
const mlApi = axios.create({
  baseURL: 'http://localhost:8080/api/ml',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Ensure no global interceptors affect this instance
mlApi.interceptors.request.clear()
mlApi.interceptors.response.clear()

// Request interceptor - ML endpoints don't need authentication
mlApi.interceptors.request.use(
  (config) => {
    // ML endpoints are public - actively remove any auth headers
    delete config.headers.Authorization
    delete config.headers.authorization
    console.log('ML API Request:', config.method?.toUpperCase(), config.url, 'Auth headers removed')
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling  
mlApi.interceptors.response.use(
  (response) => {
    console.log('ML API Success:', response.status, response.config.method?.toUpperCase(), response.config.url)
    return response
  },
  (error) => {
    console.error('ML API Error:', error.response?.status, error.config?.method?.toUpperCase(), error.config?.url, error.message)
    // ML endpoints don't require authentication, so don't redirect on errors
    return Promise.reject(error)
  }
)

// ML API endpoints
export const mlApiService = {
  // Predictions
  predictions: {
    predictNeuropathyProgression: (patientId, readingId = null) => {
      const params = readingId ? { readingId } : {}
      return mlApi.post(`/predict/neuropathy-progression/${patientId}`, null, { params })
    },
    
    predictGlucoseComplications: (patientId, readingId = null) => {
      const params = readingId ? { readingId } : {}
      return mlApi.post(`/predict/glucose-complications/${patientId}`, null, { params })
    },
    
    detectAnomalies: (patientId, readingId = null) => {
      const params = readingId ? { readingId } : {}
      return mlApi.post(`/detect/anomalies/${patientId}`, null, { params })
    },
    
    calculateRiskStratification: (patientId, readingId = null) => {
      const params = readingId ? { readingId } : {}
      return mlApi.post(`/predict/risk-stratification/${patientId}`, null, { params })
    },
    
    getPatientPredictions: (patientId, modelType = null) => {
      const params = modelType ? { modelType } : {}
      return mlApi.get(`/predictions/${patientId}`, { params })
    },
    
    getRecentPredictions: (limit = 10) => {
      return mlApi.get('/predictions/recent', { params: { limit } })
    }
  },

  // Models
  models: {
    getAvailableModels: () => mlApi.get('/models'),
    getModelMetrics: (modelType) => mlApi.get(`/metrics/${modelType}`)
  },

  // Health
  health: {
    checkHealth: () => mlApi.get('/health')
  }
}

export default mlApiService