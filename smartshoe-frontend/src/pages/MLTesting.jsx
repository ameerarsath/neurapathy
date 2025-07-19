import React, { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { mlApiService } from '../services/mlApi'
import { smartShoeAPI } from '../services/api'
import { useAuth } from '../contexts/AuthContext'
import {
  Brain,
  Activity,
  AlertTriangle,
  TrendingUp,
  Zap,
  Users,
  TestTube,
  CheckCircle,
  XCircle,
  RefreshCw,
  Play,
  Settings,
  BarChart3,
  Info,
  Heart,
  Thermometer,
  Waves
} from 'lucide-react'
import LoadingSpinner from '../components/common/LoadingSpinner'
import toast from 'react-hot-toast'

const MLTesting = () => {
  const { user, canAccess } = useAuth()
  const [selectedPatient, setSelectedPatient] = useState('')
  const [selectedReading, setSelectedReading] = useState('')
  const [testResults, setTestResults] = useState({})
  const [isRunningTests, setIsRunningTests] = useState(false)
  const [mlHealthStatus, setMlHealthStatus] = useState(null)

  // Fetch patients
  const { data: patients = [] } = useQuery({
    queryKey: ['patients'],
    queryFn: () => smartShoeAPI.patients.getAll(),
    enabled: canAccess('PROVIDER'),
    select: data => data?.data?.patients || data?.data || []
  })

  // Fetch readings for selected patient
  const { data: readings = [] } = useQuery({
    queryKey: ['patient-readings', selectedPatient],
    queryFn: () => smartShoeAPI.medicalReadings.getByPatient(selectedPatient),
    enabled: canAccess('PROVIDER') && !!selectedPatient,
    select: data => data?.data?.readings || data?.data || []
  })

  // Fetch existing ML predictions
  const { data: predictions = [], refetch: refetchPredictions } = useQuery({
    queryKey: ['ml-predictions', selectedPatient],
    queryFn: () => mlApiService.predictions.getPatientPredictions(selectedPatient),
    enabled: canAccess('PROVIDER') && !!selectedPatient,
    select: data => data?.data?.predictions || []
  })

  // Check ML service health
  const healthQuery = useQuery({
    queryKey: ['ml-health'],
    queryFn: () => mlApiService.health.checkHealth(),
    refetchInterval: 30000, // Check every 30 seconds
    onSuccess: (data) => setMlHealthStatus(data?.data),
    onError: () => setMlHealthStatus({ status: 'unhealthy', error: 'Service unavailable' })
  })

  // Test mutations
  const testNeuropathy = useMutation({
    mutationFn: () => mlApiService.predictions.predictNeuropathyProgression(selectedPatient, selectedReading || null),
    onSuccess: (data) => {
      const result = data?.data
      setTestResults(prev => ({
        ...prev,
        neuropathy: { success: true, data: result, timestamp: new Date().toISOString() }
      }))
      toast.success('Neuropathy test completed')
      refetchPredictions()
    },
    onError: (error) => {
      setTestResults(prev => ({
        ...prev,
        neuropathy: { success: false, error: error.message, timestamp: new Date().toISOString() }
      }))
      toast.error('Neuropathy test failed')
    }
  })

  const testGlucose = useMutation({
    mutationFn: () => mlApiService.predictions.predictGlucoseComplications(selectedPatient, selectedReading || null),
    onSuccess: (data) => {
      const result = data?.data
      setTestResults(prev => ({
        ...prev,
        glucose: { success: true, data: result, timestamp: new Date().toISOString() }
      }))
      toast.success('Glucose test completed')
      refetchPredictions()
    },
    onError: (error) => {
      setTestResults(prev => ({
        ...prev,
        glucose: { success: false, error: error.message, timestamp: new Date().toISOString() }
      }))
      toast.error('Glucose test failed')
    }
  })

  const testAnomaly = useMutation({
    mutationFn: () => mlApiService.predictions.detectAnomalies(selectedPatient, selectedReading || null),
    onSuccess: (data) => {
      const result = data?.data
      setTestResults(prev => ({
        ...prev,
        anomaly: { success: true, data: result, timestamp: new Date().toISOString() }
      }))
      toast.success('Anomaly test completed')
      refetchPredictions()
    },
    onError: (error) => {
      setTestResults(prev => ({
        ...prev,
        anomaly: { success: false, error: error.message, timestamp: new Date().toISOString() }
      }))
      toast.error('Anomaly test failed')
    }
  })

  const testRisk = useMutation({
    mutationFn: () => mlApiService.predictions.calculateRiskStratification(selectedPatient, selectedReading || null),
    onSuccess: (data) => {
      const result = data?.data
      setTestResults(prev => ({
        ...prev,
        risk: { success: true, data: result, timestamp: new Date().toISOString() }
      }))
      toast.success('Risk test completed')
      refetchPredictions()
    },
    onError: (error) => {
      setTestResults(prev => ({
        ...prev,
        risk: { success: false, error: error.message, timestamp: new Date().toISOString() }
      }))
      toast.error('Risk test failed')
    }
  })

  const runAllTests = async () => {
    if (!selectedPatient) {
      toast.error('Please select a patient first')
      return
    }

    setIsRunningTests(true)
    setTestResults({})

    try {
      await Promise.allSettled([
        testNeuropathy.mutateAsync(),
        testGlucose.mutateAsync(),
        testAnomaly.mutateAsync(),
        testRisk.mutateAsync()
      ])
      toast.success('All tests completed!')
    } catch (error) {
      toast.error('Some tests failed. Check individual results.')
    } finally {
      setIsRunningTests(false)
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'unhealthy':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <RefreshCw className="h-5 w-5 text-gray-500 animate-spin" />
    }
  }

  const getResultIcon = (result) => {
    if (!result) return <div className="h-5 w-5" />
    return result.success ? 
      <CheckCircle className="h-5 w-5 text-green-500" /> : 
      <XCircle className="h-5 w-5 text-red-500" />
  }

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'HIGH':
        return 'bg-red-100 text-red-800'
      case 'MEDIUM':
        return 'bg-yellow-100 text-yellow-800'
      case 'LOW':
        return 'bg-green-100 text-green-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  if (!canAccess('PROVIDER')) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Access Denied</h1>
          <p className="text-gray-600">ML testing requires provider privileges</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center">
                <TestTube className="h-6 w-6 text-primary-600 mr-2" />
                ML Model Testing Lab
              </h1>
              <p className="text-gray-600 mt-1">
                Test and validate machine learning models for neuropathy detection
              </p>
            </div>
            <div className="flex items-center space-x-2">
              {getStatusIcon(mlHealthStatus?.status)}
              <span className={`text-sm font-medium ${
                mlHealthStatus?.status === 'healthy' ? 'text-green-600' : 'text-red-600'
              }`}>
                ML Service: {mlHealthStatus?.status || 'Checking...'}
              </span>
            </div>
          </div>
        </div>

        {/* Test Configuration */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Settings className="h-5 w-5 text-gray-500 mr-2" />
            Test Configuration
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Patient Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Patient
              </label>
              <select
                value={selectedPatient}
                onChange={(e) => {
                  setSelectedPatient(e.target.value)
                  setSelectedReading('')
                  setTestResults({})
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="">Choose a patient...</option>
                {patients.map(patient => (
                  <option key={patient.id} value={patient.id}>
                    {patient.firstName} {patient.lastName} (ID: {patient.id})
                  </option>
                ))}
              </select>
              {selectedPatient && (
                <p className="text-sm text-gray-500 mt-1">
                  Selected: {patients.find(p => p.id == selectedPatient)?.firstName} {patients.find(p => p.id == selectedPatient)?.lastName}
                </p>
              )}
            </div>

            {/* Reading Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Reading (Optional)
              </label>
              <select
                value={selectedReading}
                onChange={(e) => setSelectedReading(e.target.value)}
                disabled={!selectedPatient}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:bg-gray-100"
              >
                <option value="">Use latest reading...</option>
                {readings.map(reading => (
                  <option key={reading.id} value={reading.id}>
                    {reading.readingType} - {new Date(reading.recordedAt).toLocaleDateString()}
                  </option>
                ))}
              </select>
              {selectedReading && (
                <p className="text-sm text-gray-500 mt-1">
                  Selected: {readings.find(r => r.id == selectedReading)?.readingType} reading
                </p>
              )}
            </div>
          </div>

          {/* Test Actions */}
          <div className="mt-6 flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={runAllTests}
                disabled={!selectedPatient || isRunningTests}
                className="flex items-center px-6 py-3 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
              >
                {isRunningTests ? (
                  <RefreshCw className="h-5 w-5 mr-2 animate-spin" />
                ) : (
                  <Play className="h-5 w-5 mr-2" />
                )}
                Run All Tests
              </button>
              
              <button
                onClick={() => healthQuery.refetch()}
                className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Check Health
              </button>
            </div>
            
            <div className="text-sm text-gray-500">
              {selectedPatient ? `Testing with Patient ID: ${selectedPatient}` : 'No patient selected'}
            </div>
          </div>
        </div>

        {/* Individual Test Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <button
            onClick={() => testNeuropathy.mutate()}
            disabled={!selectedPatient || testNeuropathy.isLoading}
            className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center justify-between mb-2">
              <Brain className="h-6 w-6 text-blue-600" />
              {getResultIcon(testResults.neuropathy)}
            </div>
            <p className="font-medium text-gray-900">Neuropathy Test</p>
            <p className="text-sm text-gray-600">Progression analysis</p>
            {testNeuropathy.isLoading && <div className="mt-2 text-xs text-blue-600">Testing...</div>}
          </button>

          <button
            onClick={() => testGlucose.mutate()}
            disabled={!selectedPatient || testGlucose.isLoading}
            className="p-4 bg-green-50 hover:bg-green-100 rounded-lg text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center justify-between mb-2">
              <Activity className="h-6 w-6 text-green-600" />
              {getResultIcon(testResults.glucose)}
            </div>
            <p className="font-medium text-gray-900">Glucose Test</p>
            <p className="text-sm text-gray-600">Complications risk</p>
            {testGlucose.isLoading && <div className="mt-2 text-xs text-green-600">Testing...</div>}
          </button>

          <button
            onClick={() => testAnomaly.mutate()}
            disabled={!selectedPatient || testAnomaly.isLoading}
            className="p-4 bg-yellow-50 hover:bg-yellow-100 rounded-lg text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center justify-between mb-2">
              <Zap className="h-6 w-6 text-yellow-600" />
              {getResultIcon(testResults.anomaly)}
            </div>
            <p className="font-medium text-gray-900">Anomaly Test</p>
            <p className="text-sm text-gray-600">Sensor analysis</p>
            {testAnomaly.isLoading && <div className="mt-2 text-xs text-yellow-600">Testing...</div>}
          </button>

          <button
            onClick={() => testRisk.mutate()}
            disabled={!selectedPatient || testRisk.isLoading}
            className="p-4 bg-purple-50 hover:bg-purple-100 rounded-lg text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center justify-between mb-2">
              <TrendingUp className="h-6 w-6 text-purple-600" />
              {getResultIcon(testResults.risk)}
            </div>
            <p className="font-medium text-gray-900">Risk Test</p>
            <p className="text-sm text-gray-600">Stratification</p>
            {testRisk.isLoading && <div className="mt-2 text-xs text-purple-600">Testing...</div>}
          </button>
        </div>

        {/* Test Results */}
        {Object.keys(testResults).length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <BarChart3 className="h-5 w-5 text-gray-500 mr-2" />
              Test Results
            </h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {Object.entries(testResults).map(([testType, result]) => (
                <div key={testType} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-medium text-gray-900 capitalize">{testType} Test</h3>
                    {getResultIcon(result)}
                  </div>
                  
                  {result.success ? (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Risk Level:</span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(result.data.risk_level)}`}>
                          {result.data.risk_level}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Prediction:</span>
                        <span className="text-sm font-medium text-gray-900">
                          {(result.data.prediction.prediction * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Confidence:</span>
                        <span className="text-sm font-medium text-gray-900">
                          {(result.data.prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      {result.data.prediction.additionalData && (
                        <div className="bg-gray-50 rounded p-3 mt-3">
                          <p className="text-xs font-medium text-gray-700 mb-2">Additional Data:</p>
                          <pre className="text-xs text-gray-600 overflow-x-auto">
                            {JSON.stringify(result.data.prediction.additionalData, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-sm text-red-600">
                      Error: {result.error}
                    </div>
                  )}
                  
                  <div className="text-xs text-gray-500 mt-3">
                    {new Date(result.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Historical Predictions */}
        {predictions.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Brain className="h-5 w-5 text-gray-500 mr-2" />
              Historical Predictions
            </h2>
            
            <div className="space-y-4">
              {predictions.slice(0, 5).map((prediction, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-gray-900 capitalize">
                      {prediction.modelType.replace('_', ' ')}
                    </h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(prediction.riskLevel)}`}>
                      {prediction.riskLevel}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Prediction:</span>
                      <p className="font-medium">{(prediction.predictionValue * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Confidence:</span>
                      <p className="font-medium">{(prediction.confidence * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Date:</span>
                      <p className="font-medium">{new Date(prediction.timestamp).toLocaleDateString()}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default MLTesting